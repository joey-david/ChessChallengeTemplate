"""
Reinforcement learning fine-tuning with Stockfish evaluations as rewards.

This script runs simple policy-gradient updates using Stockfish score
evaluations after the model's moves. It keeps evaluation compatibility
by operating on move-level tokens.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.utils import convert_extended_uci_to_uci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL fine-tuning with Stockfish rewards")

    parser.add_argument("--model_path", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--output_dir", type=str, default="./rl_output", help="Output directory")
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--max_moves", type=int, default=120, help="Max moves per game")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--stockfish_path", type=str, default=None, help="Stockfish executable path")
    parser.add_argument("--stockfish_level", type=int, default=1, help="Stockfish skill level (0-20)")
    parser.add_argument("--eval_time", type=float, default=0.05, help="Stockfish eval time per move")
    parser.add_argument("--illegal_penalty", type=float, default=-1.0, help="Penalty for illegal moves")
    parser.add_argument(
        "--auto_level",
        action="store_true",
        default=True,
        help="Automatically increase Stockfish level on sustained wins",
    )
    parser.add_argument(
        "--no_auto_level",
        action="store_false",
        dest="auto_level",
        help="Disable automatic Stockfish level increases",
    )
    parser.add_argument("--auto_level_window", type=int, default=10, help="Games per winrate window")
    parser.add_argument("--auto_level_threshold", type=float, default=0.5, help="Winrate threshold")
    parser.add_argument("--auto_level_step", type=int, default=1, help="Stockfish level step")
    parser.add_argument(
        "--model_color",
        type=str,
        default="alternate",
        choices=["white", "black", "alternate"],
        help="Which side the model plays",
    )
    parser.add_argument("--save_every", type=int, default=25, help="Save every N episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def _convert_board_to_moves(board) -> str:
    import chess

    moves = []
    temp_board = chess.Board()

    for move in board.move_stack:
        color = "W" if temp_board.turn == chess.WHITE else "B"
        piece = temp_board.piece_at(move.from_square)
        piece_letter = piece.symbol().upper() if piece else "P"

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        move_str = f"{color}{piece_letter}{from_sq}{to_sq}"

        if move.promotion:
            move_str += f"={chess.piece_symbol(move.promotion).upper()}"

        if temp_board.is_capture(move):
            move_str += "(x)"

        temp_board.push(move)
        if temp_board.is_checkmate():
            move_str = move_str.replace("(x)", "(x+*)") if "(x)" in move_str else move_str + "(+*)"
        elif temp_board.is_check():
            move_str = move_str.replace("(x)", "(x+)") if "(x)" in move_str else move_str + "(+)"

        if piece_letter == "K" and abs(ord(from_sq[0]) - ord(to_sq[0])) > 1:
            if to_sq[0] == "g":
                move_str = move_str.split("(")[0] + "(o)"
            else:
                move_str = move_str.split("(")[0] + "(O)"

        moves.append(move_str)

    return " ".join(moves)


def _build_input_text(board, tokenizer) -> str:
    moves_str = _convert_board_to_moves(board)
    if not moves_str:
        return tokenizer.bos_token
    return tokenizer.bos_token + " " + moves_str


def _score_board(engine, board, model_color: str, eval_time: float) -> float:
    import chess
    import chess.engine

    limit = chess.engine.Limit(time=eval_time)
    info = engine.analyse(board, limit)
    pov = chess.WHITE if model_color == "white" else chess.BLACK
    score = info["score"].pov(pov)
    cp = score.score(mate_score=10000)
    if cp is None:
        return 0.0

    scaled = cp / 1000.0
    return max(min(scaled, 1.0), -1.0)


def _sample_model_move(
    model,
    tokenizer,
    board,
    device: torch.device,
    temperature: float,
) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    import chess

    input_text = _build_input_text(board, tokenizer)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=model.config.n_ctx - 1,
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits[:, -1, :] / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    next_token = torch.multinomial(probs, num_samples=1)
    log_prob = log_probs[0, next_token.item()]

    move_token = tokenizer.decode(next_token[0])
    if len(move_token) < 6:
        return None, log_prob

    uci_move = convert_extended_uci_to_uci(move_token)
    try:
        move = chess.Move.from_uci(uci_move)
    except (ValueError, chess.InvalidMoveError):
        return None, log_prob

    if move not in board.legal_moves:
        return None, log_prob

    return uci_move, log_prob


def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def _episode_outcome(
    board,
    model_color: str,
    illegal_end: bool,
    max_moves_reached: bool,
) -> str:
    if illegal_end:
        return "loss"

    if board.is_game_over():
        result = board.result()
        if result == "1/2-1/2":
            return "draw"
        if result == "1-0":
            return "win" if model_color == "white" else "loss"
        if result == "0-1":
            return "win" if model_color == "black" else "loss"
        return "draw"

    if max_moves_reached:
        return "draw"

    return "draw"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM
    from src.tokenizer import ChessTokenizer
    from src.model import ChessConfig, ChessForCausalLM

    import chess
    import chess.engine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ChessTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.stockfish_path is None:
        import shutil

        args.stockfish_path = shutil.which("stockfish")
    if not args.stockfish_path:
        raise RuntimeError("Stockfish not found. Provide --stockfish_path.")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    current_level = args.stockfish_level
    engine.configure({"Skill Level": current_level})

    print("=" * 60)
    print("CHESS CHALLENGE - RL FINE-TUNING")
    print("=" * 60)

    recent_results: Deque[str] = deque(maxlen=args.auto_level_window)

    for episode in range(1, args.num_episodes + 1):
        model.train()
        board = chess.Board()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        illegal_end = False

        if args.model_color == "alternate":
            model_color = "white" if episode % 2 == 1 else "black"
        else:
            model_color = args.model_color

        while not board.is_game_over() and len(log_probs) < args.max_moves:
            is_model_turn = (board.turn == chess.WHITE) == (model_color == "white")
            if is_model_turn:
                uci_move, log_prob = _sample_model_move(
                    model=model,
                    tokenizer=tokenizer,
                    board=board,
                    device=device,
                    temperature=args.temperature,
                )

                if uci_move is None or log_prob is None:
                    rewards.append(args.illegal_penalty)
                    illegal_end = True
                    break

                board.push(chess.Move.from_uci(uci_move))
                reward = _score_board(engine, board, model_color, args.eval_time)
                log_probs.append(log_prob)
                rewards.append(reward)
            else:
                result = engine.play(board, chess.engine.Limit(time=args.eval_time))
                board.push(result.move)

        max_moves_reached = len(log_probs) >= args.max_moves and not board.is_game_over()
        outcome = _episode_outcome(board, model_color, illegal_end, max_moves_reached)
        recent_results.append(outcome)

        if log_probs:
            returns = _discounted_returns(rewards, args.gamma).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_prob_tensor = torch.stack(log_probs)
            loss = -(log_prob_tensor * returns).sum()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        else:
            loss = torch.tensor(0.0)

        if args.auto_level and len(recent_results) == args.auto_level_window:
            wins = sum(1 for result in recent_results if result == "win")
            winrate = wins / args.auto_level_window
            if winrate > args.auto_level_threshold and current_level < 20:
                current_level = min(20, current_level + args.auto_level_step)
                engine.configure({"Skill Level": current_level})
                recent_results.clear()
                print(f"Auto-level: raised Stockfish to level {current_level}")

        if episode % 10 == 0 or episode == 1:
            avg_reward = sum(rewards) / max(len(rewards), 1)
            print(
                f"Episode {episode}/{args.num_episodes} | "
                f"Moves: {len(log_probs)} | "
                f"Avg reward: {avg_reward:.3f} | "
                f"Loss: {loss.item():.4f}"
            )

        if args.save_every and episode % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-episode-{episode}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    engine.quit()

    print("\nRL fine-tuning complete.")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()
