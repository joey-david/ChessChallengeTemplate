"""
GRPO-based reinforcement learning with Stockfish move evaluations.

This script fine-tunes a trained baseline by sampling moves from the model,
scoring them with Stockfish, and applying group-relative policy optimization
to reduce illegal moves and improve playing strength.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import deque
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning with Stockfish rewards")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the baseline model or Hugging Face model ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rl_output",
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # GRPO / optimization settings
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of positions per step")
    parser.add_argument("--group_size", type=int, default=4, help="Samples per position")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta_kl", type=float, default=0.02, help="KL penalty coefficient")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping")

    # Sampling settings
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling (0 disables)")
    parser.add_argument(
        "--max_move_tokens",
        type=int,
        default=20,
        help="Max tokens to generate for a single move",
    )

    # Stockfish settings
    parser.add_argument("--stockfish_path", type=str, default=None, help="Path to Stockfish")
    parser.add_argument("--stockfish_level", type=int, default=5, help="Stockfish skill (0-20)")
    parser.add_argument(
        "--stockfish_level_max",
        type=int,
        default=None,
        help="Max Stockfish skill to ramp up to (enables auto progression)",
    )
    parser.add_argument(
        "--stockfish_level_every_seconds",
        type=int,
        default=0,
        help="Increase Stockfish skill by 1 every N seconds (0 disables)",
    )
    parser.add_argument(
        "--stockfish_level_loss_rate",
        type=float,
        default=None,
        help="Increase Stockfish skill when recent loss rate is below this value (0-1).",
    )
    parser.add_argument(
        "--stockfish_level_window",
        type=int,
        default=256,
        help="Number of recent samples to use for loss-rate ramping.",
    )
    parser.add_argument(
        "--engine_time",
        type=float,
        default=0.01,
        help="Time limit per Stockfish evaluation (seconds)",
    )

    # Reward shaping
    parser.add_argument("--illegal_penalty", type=float, default=-4.0, help="Penalty for illegal moves")
    parser.add_argument("--legal_bonus", type=float, default=0.1, help="Bonus for legal moves")
    parser.add_argument("--cp_clip", type=float, default=1000.0, help="Clip eval scores (centipawns)")
    parser.add_argument("--cp_scale", type=float, default=400.0, help="Scale eval scores (centipawns)")
    parser.add_argument(
        "--use_best_move_baseline",
        action="store_true",
        help="Reward relative to Stockfish best move for the position",
    )

    # Position sampling
    parser.add_argument("--min_random_plies", type=int, default=6, help="Min plies for random positions")
    parser.add_argument("--max_random_plies", type=int, default=30, help="Max plies for random positions")
    parser.add_argument("--max_position_retries", type=int, default=10, help="Retries for sampling positions")

    # Logging / checkpointing
    parser.add_argument("--log_steps", type=int, default=25, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=0, help="Checkpoint frequency (0 disables)")
    parser.add_argument(
        "--save_every_seconds",
        type=int,
        default=600,
        help="Checkpoint every N seconds (time-based fallback)",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, device: str):
    if "/" in model_path and not model_path.startswith("."):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Prefer custom chess tokenizers if available for local paths.
        try:
            from src.tokenizer import ChessSquaresTokenizer, ChessTokenizer

            try:
                tokenizer = ChessSquaresTokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = ChessTokenizer.from_pretrained(model_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.config.use_cache = False
    return model, tokenizer


def build_stockfish_engine(stockfish_path: Optional[str], stockfish_level: int):
    try:
        import chess
        import chess.engine
    except ImportError as exc:
        raise ImportError("python-chess is required for RL training") from exc

    if stockfish_path is None:
        import shutil

        stockfish_path = shutil.which("stockfish")

    if not stockfish_path:
        raise RuntimeError("Stockfish not found. Provide --stockfish_path.")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": stockfish_level})
    return engine, chess


SQUARE_PATTERN = r"[a-h][1-8]"


def detect_tokenizer_format(tokenizer) -> str:
    cached = getattr(tokenizer, "_cached_chess_format", None)
    if cached:
        return cached

    test_formats = {
        "decomposed": "WP e2_f e4_t",
        "standard": "WPe2e4",
        "uci": "e2e4",
        "uci_spaced": "e2 e4",
    }

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    best_format = "standard"
    min_unk_count = float("inf")

    for fmt, sample in test_formats.items():
        try:
            tokens = tokenizer.encode(sample, add_special_tokens=False)
            unk_count = tokens.count(unk_token_id) if unk_token_id is not None else 0
            if len(tokens) == 1 and unk_count == 1:
                unk_count = 100
            if unk_count < min_unk_count:
                min_unk_count = unk_count
                best_format = fmt
        except Exception:
            continue

    setattr(tokenizer, "_cached_chess_format", best_format)
    return best_format


def format_move(
    color: str,
    piece: str,
    from_sq: str,
    to_sq: str,
    promotion: Optional[str],
    tokenizer,
) -> str:
    fmt = detect_tokenizer_format(tokenizer)
    if fmt == "decomposed":
        move_str = f"{color}{piece} {from_sq}_f {to_sq}_t"
    elif fmt == "uci":
        move_str = f"{from_sq}{to_sq}"
        if promotion:
            move_str += promotion.lower()
    elif fmt == "uci_spaced":
        move_str = f"{from_sq} {to_sq}"
        if promotion:
            move_str += f" {promotion.lower()}"
    else:
        move_str = f"{color}{piece}{from_sq}{to_sq}"
        if promotion:
            move_str += f"={promotion}"
    return move_str


def convert_board_to_moves(board, chess, tokenizer) -> str:
    moves: List[str] = []
    temp_board = chess.Board()
    fmt = detect_tokenizer_format(tokenizer)

    for move in board.move_stack:
        color = "W" if temp_board.turn == chess.WHITE else "B"
        piece = temp_board.piece_at(move.from_square)
        piece_letter = piece.symbol().upper() if piece else "P"

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)

        promotion = None
        if move.promotion:
            promotion = chess.piece_symbol(move.promotion).upper()

        move_str = format_move(color, piece_letter, from_sq, to_sq, promotion, tokenizer)

        if fmt == "standard":
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
        else:
            temp_board.push(move)

        moves.append(move_str)

    return " ".join(moves)


def is_separator_token(tokenizer, token_str: str) -> bool:
    if hasattr(tokenizer, "eos_token") and token_str == tokenizer.eos_token:
        return True
    if token_str.strip() == "" and len(token_str) > 0:
        return True
    if token_str != token_str.rstrip():
        return True
    return False


def extract_uci_move(text: str) -> Optional[str]:
    if not text:
        return None

    squares = re.findall(SQUARE_PATTERN, text)
    if len(squares) < 2:
        return None

    from_sq, to_sq = squares[0], squares[1]
    uci_move = from_sq + to_sq

    to_sq_idx = text.find(to_sq)
    if to_sq_idx != -1:
        remaining = text[to_sq_idx + 2 : to_sq_idx + 5]
        promo_match = re.search(r"[=]?([qrbnQRBN])", remaining)
        if promo_match:
            uci_move += promo_match.group(1).lower()

    return uci_move


def has_complete_move(text: str) -> bool:
    squares = re.findall(SQUARE_PATTERN, text)
    return len(squares) >= 2


def sample_move_tokens(
    model,
    ref_model,
    tokenizer,
    input_ids: torch.Tensor,
    temperature: float,
    top_k: int,
    max_tokens: int,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    generated_tokens = []
    current_ids = input_ids
    accumulated_text = ""
    log_prob_sum = torch.tensor(0.0, device=input_ids.device)
    ref_log_prob_sum = torch.tensor(0.0, device=input_ids.device)

    remaining = model.config.n_ctx - current_ids.shape[1]
    max_tokens = min(max_tokens, max(0, remaining))

    for _ in range(max_tokens):
        outputs = model(input_ids=current_ids)
        logits = outputs.logits[:, -1, :]
        masked_logits = apply_top_k(logits, top_k)
        scaled_logits = masked_logits / max(temperature, 1e-5)
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = int(next_token.item())

        with torch.no_grad():
            ref_outputs = ref_model(input_ids=current_ids)
            ref_logits = ref_outputs.logits[:, -1, :]
            ref_masked_logits = apply_top_k(ref_logits, top_k)
            ref_scaled_logits = ref_masked_logits / max(temperature, 1e-5)
            ref_log_probs = torch.log_softmax(ref_scaled_logits, dim=-1)

        token_str = tokenizer.decode(next_token[0])

        if is_separator_token(tokenizer, token_str):
            if has_complete_move(accumulated_text):
                break
            if hasattr(tokenizer, "eos_token") and token_str == tokenizer.eos_token:
                break
            if accumulated_text:
                break

        generated_tokens.append(next_token[0])
        log_prob_sum = log_prob_sum + log_probs[0, token_id]
        ref_log_prob_sum = ref_log_prob_sum + ref_log_probs[0, token_id]
        current_ids = torch.cat([current_ids, next_token], dim=-1)
        accumulated_text += token_str

        if has_complete_move(accumulated_text):
            squares = re.findall(SQUARE_PATTERN, accumulated_text)
            if len(squares) >= 2:
                to_sq = squares[1]
                if to_sq[1] in "18":
                    if len(generated_tokens) > 3:
                        break
                else:
                    break

    if generated_tokens:
        all_tokens = torch.cat(generated_tokens, dim=0)
        move_str = tokenizer.decode(all_tokens, skip_special_tokens=True)
        return move_str.strip(), log_prob_sum, ref_log_prob_sum

    return "", log_prob_sum, ref_log_prob_sum


def token_to_uci(token: str) -> Optional[str]:
    if not token:
        return None
    base = token.split("(", 1)[0]
    if len(base) < 6:
        return None
    from_sq = base[2:4]
    to_sq = base[4:6]
    if (
        from_sq[0] not in "abcdefgh"
        or from_sq[1] not in "12345678"
        or to_sq[0] not in "abcdefgh"
        or to_sq[1] not in "12345678"
    ):
        return None
    promo = ""
    if "=" in base:
        promo_idx = base.index("=")
        if promo_idx + 1 < len(base):
            promo = base[promo_idx + 1].lower()
    return from_sq + to_sq + promo


def sample_random_position(chess, rng: random.Random, min_plies: int, max_plies: int, retries: int):
    for _ in range(retries):
        board = chess.Board()
        n_plies = rng.randint(min_plies, max_plies)
        for _ in range(n_plies):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
        if not board.is_game_over():
            return board
    return chess.Board()


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    topk_vals = torch.topk(logits, top_k, dim=-1).values
    threshold = topk_vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def evaluate_move_reward(
    board,
    move_uci: str,
    engine,
    chess,
    eval_time: float,
    mate_score: int,
    cp_clip: float,
    cp_scale: float,
    legal_bonus: float,
    illegal_penalty: float,
    use_best_move_baseline: bool,
) -> Tuple[float, bool, Optional[float]]:
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return illegal_penalty, False, None

    if move not in board.legal_moves:
        return illegal_penalty, False, None

    color = board.turn
    board_after = board.copy()
    board_after.push(move)

    info_after = engine.analyse(board_after, chess.engine.Limit(time=eval_time))
    score_after = info_after["score"].pov(color).score(mate_score=mate_score)
    if score_after is None:
        score_after = 0

    score_after = max(min(score_after, cp_clip), -cp_clip)
    reward = legal_bonus + (score_after / cp_scale)

    if use_best_move_baseline:
        info_best = engine.analyse(board, chess.engine.Limit(time=eval_time))
        score_best = info_best["score"].pov(color).score(mate_score=mate_score)
        if score_best is None:
            score_best = 0
        score_best = max(min(score_best, cp_clip), -cp_clip)
        reward = legal_bonus + ((score_after - score_best) / cp_scale)

    return reward, True, score_after


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    if (
        args.stockfish_level_max is not None
        and args.stockfish_level_every_seconds <= 0
        and args.stockfish_level_loss_rate is None
    ):
        raise ValueError(
            "--stockfish_level_max requires --stockfish_level_every_seconds > 0 "
            "or --stockfish_level_loss_rate"
        )
    if args.stockfish_level_loss_rate is not None:
        if not (0.0 < args.stockfish_level_loss_rate < 1.0):
            raise ValueError("--stockfish_level_loss_rate must be between 0 and 1")
        if args.stockfish_level_window <= 0:
            raise ValueError("--stockfish_level_window must be > 0")
        if args.stockfish_level_every_seconds > 0:
            raise ValueError("--stockfish_level_loss_rate is incompatible with --stockfish_level_every_seconds")

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    ref_model, _ = load_model_and_tokenizer(args.model_path, args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    current_stockfish_level = max(0, min(20, args.stockfish_level))
    max_stockfish_level = (
        max(0, min(20, args.stockfish_level_max))
        if args.stockfish_level_max is not None
        else current_stockfish_level
    )
    engine, chess = build_stockfish_engine(args.stockfish_path, current_stockfish_level)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    tokenizer.truncation_side = "left"

    total_samples = 0
    total_illegal = 0
    running_reward = 0.0

    max_length = max(1, model.config.n_ctx - max(args.max_move_tokens, 1) - 1)

    start_time = time.time()
    last_save_time = start_time
    next_level_update = (
        start_time + args.stockfish_level_every_seconds
        if args.stockfish_level_every_seconds > 0 and current_stockfish_level < max_stockfish_level
        else None
    )
    loss_window = (
        deque(maxlen=args.stockfish_level_window)
        if args.stockfish_level_loss_rate is not None and current_stockfish_level < max_stockfish_level
        else None
    )

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        step_samples = 0
        step_illegal = 0
        step_reward = 0.0

        for _ in range(args.grad_accum_steps):
            boards = [
                sample_random_position(
                    chess, rng, args.min_random_plies, args.max_random_plies, args.max_position_retries
                )
                for _ in range(args.batch_size)
            ]

            prompts = []
            for board in boards:
                moves_str = convert_board_to_moves(board, chess, tokenizer)
                if moves_str:
                    prompts.append(f"{tokenizer.bos_token} {moves_str}")
                else:
                    prompts.append(tokenizer.bos_token)

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            sample_log_probs_rows = []
            sample_ref_log_probs_rows = []
            sample_rewards = torch.zeros(args.batch_size, args.group_size, device="cpu")
            sample_illegal = torch.zeros_like(sample_rewards, device="cpu")

            for i, board in enumerate(boards):
                seq_len = int(attention_mask[i].sum().item())
                prompt_ids = input_ids[i : i + 1, :seq_len]
                row_log_probs = []
                row_ref_log_probs = []
                for j in range(args.group_size):
                    move_text, log_prob_sum, ref_log_prob_sum = sample_move_tokens(
                        model=model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        input_ids=prompt_ids,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        max_tokens=args.max_move_tokens,
                    )
                    move_uci = extract_uci_move(move_text)
                    if move_uci is None:
                        reward = args.illegal_penalty
                        legal = False
                        score_after = None
                    else:
                        reward, legal, score_after = evaluate_move_reward(
                            board=board,
                            move_uci=move_uci,
                            engine=engine,
                            chess=chess,
                            eval_time=args.engine_time,
                            mate_score=int(args.cp_clip),
                            cp_clip=args.cp_clip,
                            cp_scale=args.cp_scale,
                            legal_bonus=args.legal_bonus,
                            illegal_penalty=args.illegal_penalty,
                            use_best_move_baseline=args.use_best_move_baseline,
                        )
                    sample_rewards[i, j] = reward
                    sample_illegal[i, j] = 0.0 if legal else 1.0
                    row_log_probs.append(log_prob_sum)
                    row_ref_log_probs.append(ref_log_prob_sum)
                    if loss_window is not None:
                        is_loss = (not legal) or (score_after is None) or (score_after < 0)
                        loss_window.append(1.0 if is_loss else 0.0)
                sample_log_probs_rows.append(torch.stack(row_log_probs))
                sample_ref_log_probs_rows.append(torch.stack(row_ref_log_probs))

            sample_log_probs = torch.stack(sample_log_probs_rows)
            sample_ref_log_probs = torch.stack(sample_ref_log_probs_rows)

            advantages = torch.zeros_like(sample_rewards)
            for i in range(args.batch_size):
                rewards = sample_rewards[i]
                mean = rewards.mean()
                std = rewards.std(unbiased=False)
                if std > 1e-6:
                    advantages[i] = (rewards - mean) / std
                else:
                    advantages[i] = rewards - mean

            advantages = advantages.to(sample_log_probs.device)
            rewards_device = sample_rewards.to(sample_log_probs.device)
            illegal_device = sample_illegal.to(sample_log_probs.device)

            loss = -(
                advantages * sample_log_probs
                - args.beta_kl * (sample_log_probs - sample_ref_log_probs)
            ).mean()

            (loss / args.grad_accum_steps).backward()

            step_loss += loss.item()
            step_samples += args.batch_size * args.group_size
            step_illegal += int(illegal_device.sum().item())
            step_reward += float(rewards_device.sum().item())

        if next_level_update is not None:
            now = time.time()
            if now >= next_level_update:
                while next_level_update <= now and current_stockfish_level < max_stockfish_level:
                    current_stockfish_level += 1
                    engine.configure({"Skill Level": current_stockfish_level})
                    next_level_update += args.stockfish_level_every_seconds
                print(f"updated stockfish_level={current_stockfish_level}")
        if loss_window is not None and len(loss_window) == loss_window.maxlen:
            loss_rate = sum(loss_window) / len(loss_window)
            if loss_rate <= args.stockfish_level_loss_rate and current_stockfish_level < max_stockfish_level:
                current_stockfish_level += 1
                engine.configure({"Skill Level": current_stockfish_level})
                loss_window.clear()
                print(
                    f"updated stockfish_level={current_stockfish_level} "
                    f"(loss_rate={loss_rate:.2%})"
                )

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        total_samples += step_samples
        total_illegal += step_illegal
        running_reward += step_reward

        if step % args.log_steps == 0:
            avg_reward = step_reward / max(step_samples, 1)
            illegal_rate = step_illegal / max(step_samples, 1)
            print(
                f"step={step} loss={step_loss:.4f} avg_reward={avg_reward:.3f} "
                f"illegal_rate={illegal_rate:.2%}"
            )

        should_save_step = args.save_steps > 0 and step % args.save_steps == 0
        should_save_time = (time.time() - last_save_time) >= args.save_every_seconds
        if should_save_step or should_save_time:
            save_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            last_save_time = time.time()

    avg_reward = running_reward / max(total_samples, 1)
    illegal_rate = total_illegal / max(total_samples, 1)
    print(f"final avg_reward={avg_reward:.3f} illegal_rate={illegal_rate:.2%}")

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    engine.quit()


if __name__ == "__main__":
    main()
