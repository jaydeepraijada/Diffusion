import argparse
import textwrap
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForMaskedLM
from safetensors.torch import load_file
from tokenizer import get_tokenizer


FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:/Windows/Fonts/consola.ttf",
]

def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def decode_for_display(input_tokens, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_tokens[0])
    cleaned = []
    for tok in tokens:
        if tok == tokenizer.mask_token:
            cleaned.append("[MASK]")
        elif tok in tokenizer.all_special_tokens:
            continue
        else:
            cleaned.append(tok)
    return tokenizer.convert_tokens_to_string(cleaned)


def render_frame(text, step, total_steps, width=900, height=320, font_size=14, padding=24):
    BG         = (13,  17,  23)
    FG         = (230, 237, 243)
    MASK_COLOR = (88,  166, 255)
    DIM        = (110, 118, 129)

    img  = Image.new("RGB", (width, height), color=BG)
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)

    # Header
    header = f"Diffusion LM  ·  Step {step} / {total_steps}"
    draw.text((padding, padding), header, fill=DIM, font=font)
    draw.line(
        [(padding, padding + font_size + 8), (width - padding, padding + font_size + 8)],
        fill=DIM, width=1,
    )

    # Body — word-by-word coloring so [MASK] tokens appear in blue
    char_w = font_size // 2 + 2
    chars_per_line = max(1, (width - 2 * padding) // char_w)
    wrapped_lines  = textwrap.wrap(text, width=chars_per_line) or [""]

    y = padding + font_size + 18
    for line in wrapped_lines[:7]:
        x = padding
        for word in line.split():
            color   = MASK_COLOR if "[MASK]" in word else FG
            display = word + " "
            try:
                bbox = draw.textbbox((x, y), display, font=font)
                w    = bbox[2] - bbox[0]
            except AttributeError:
                w = len(display) * char_w
            if x + w > width - padding:
                x  = padding
                y += font_size + 4
            draw.text((x, y), display, fill=color, font=font)
            x += w
        y += font_size + 4
        if y > height - padding - 20:
            break

    # Progress bar
    bar_y = height - padding - 6
    bar_w = width  - 2 * padding
    draw.rectangle([padding, bar_y, padding + bar_w, bar_y + 5], fill=(30, 40, 50))
    filled = int(bar_w * step / max(total_steps, 1))
    if filled > 0:
        draw.rectangle([padding, bar_y, padding + filled, bar_y + 5], fill=MASK_COLOR)

    return img


@torch.no_grad()
def run_and_collect_frames(input_tokens, mask, attention_mask,
                            model, tokenizer, num_steps,
                            remasking="low_confidence", device="cuda",
                            frame_every=5):
    frames = []
    times  = torch.linspace(1, 0, num_steps + 1, device=device)

    for i, (t, s) in enumerate(zip(times[:-1], times[1:])):
        logits = model(input_tokens, attention_mask=attention_mask).logits

        probs = torch.softmax(logits[mask], dim=-1)
        input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        if remasking == "random":
            rp   = torch.rand_like(mask, dtype=torch.float)
            mask = mask & (rp < s / t)
            input_tokens[mask] = tokenizer.mask_token_id

        elif remasking == "low_confidence":
            probs_all    = torch.softmax(logits, dim=-1)
            chosen_probs = torch.gather(probs_all, -1, input_tokens.unsqueeze(-1)).squeeze(-1)
            chosen_probs[~mask] = 1.0
            num_to_remask = int((s / t) * mask.sum().item())
            if num_to_remask > 0:
                idx      = torch.topk(chosen_probs, num_to_remask, largest=False).indices
                new_mask = torch.zeros_like(mask)
                new_mask[0, idx] = True
                mask = new_mask
                input_tokens[mask] = tokenizer.mask_token_id

        if i % frame_every == 0 or i == num_steps - 1:
            text = decode_for_display(input_tokens, tokenizer)
            frames.append((i + 1, text))

    final_text = tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]
    return frames, final_text


def create_gif(frames, total_steps, output_path, fps=8):
    images = [render_frame(text, step, total_steps) for step, text in frames]
    if not images:
        return
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF → {output_path}  ({len(images)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create inference GIF")
    parser.add_argument("--safetensors_path", required=True,  type=str)
    parser.add_argument("--hf_model_name",    default="answerdotai/ModernBERT-base", type=str)
    parser.add_argument("--prompt",           default=None,   type=str)
    parser.add_argument("--seq_len",          default=128,    type=int)
    parser.add_argument("--num_steps",        default=64,     type=int)
    parser.add_argument("--strategy",         default="low_confidence",
                        choices=["random", "low_confidence"])
    parser.add_argument("--frame_every",      default=4,      type=int,
                        help="Capture a GIF frame every N inference steps")
    parser.add_argument("--fps",              default=8,      type=int)
    parser.add_argument("--output",           default="inference.gif", type=str)
    parser.add_argument("--device",           default="cuda", type=str)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.hf_model_name)
    model     = AutoModelForMaskedLM.from_pretrained(args.hf_model_name, device_map=args.device)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(load_file(args.safetensors_path), strict=False)
    model.tie_weights()
    model.eval()

    if args.prompt is None:
        input_tokens = torch.full((1, args.seq_len), tokenizer.mask_token_id,
                                   dtype=torch.long, device=args.device)
        mask = torch.ones((1, args.seq_len), dtype=torch.bool, device=args.device)
    else:
        chat       = [{"role": "user", "content": args.prompt}]
        prompt_ids = tokenizer.apply_chat_template(chat, tokenize=True,
                                                    add_special_tokens=True,
                                                    add_generation_prompt=True)
        input_tokens = torch.full((1, args.seq_len), tokenizer.mask_token_id,
                                   dtype=torch.long, device=args.device)
        mask = torch.ones((1, args.seq_len), dtype=torch.bool, device=args.device)
        pt   = torch.tensor(prompt_ids, device=args.device)
        input_tokens[0, :len(pt)] = pt
        mask[0, :len(pt)]         = False

    attention_mask = torch.ones((1, args.seq_len), dtype=torch.long, device=args.device)

    print("Running inference...")
    frames, final_text = run_and_collect_frames(
        input_tokens, mask, attention_mask,
        model, tokenizer, args.num_steps,
        remasking=args.strategy, device=args.device,
        frame_every=args.frame_every,
    )

    print("Final text:", final_text)
    create_gif(frames, args.num_steps, args.output, fps=args.fps)
