import os
import csv
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from evaluation.slr_eval.wer_calculation import evaluate


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group["lr"] for group in optimizer.optimizer.param_groups]

    for batch_idx, data in enumerate(tqdm(loader)):
        data = device.dict_data_to_device(data)
        ret_dict = model(data)

        loss, loss_details = model.get_loss(ret_dict, data)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data["origin_info"])
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                f"\tEpoch: {epoch_idx}, Batch({batch_idx}/{len(loader)}) done. "
                f"Loss: {loss.item():.2f}  lr:{clr[0]:.6f}"
            )
            recoder.print_log(
                "\t" + ", ".join([f"{k}: {v.item():.2f}" for k, v in loss_details.items()])
            )
    if hasattr(optimizer, 'scheduler') and hasattr(optimizer.scheduler, 'step_epoch'):
        optimizer.scheduler.step_epoch(epoch_idx)
    else:
        optimizer.scheduler.step()
    #optimizer.scheduler.step()
    recoder.print_log("\tMean training loss: {:.10f}.".format(np.mean(loss_value)))
    return loss_value


def ctm_to_word_dict(ctm_file: str):
    """Parse CTM into {id: [w1, w2, ...]}."""
    out = {}
    with open(ctm_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # CTM: <id> 1 <start> <dur> <token>
            if len(parts) >= 5:
                vid = parts[0]
                tok = parts[4]
                out.setdefault(vid, []).append(tok)
    return out


def write_csv_all_ids(csv_file: str, all_ids: list, word_dict: dict):
    """Write 1 row per id in all_ids; empty gloss if id not in word_dict."""
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "gloss"])
        for vid in all_ids:
            words = word_dict.get(vid, [])
            w.writerow([vid, " ".join(words)])


def get_split_order_ids(cfg, task: str, mode: str):
    """
    Return IDs in the exact order of the official split file used by this repo.

    Repo convention (MSLR):
        ./datasets/mslr/{task}_{mode}_info.json
    where task is "us" or "si", mode is "dev" or "test".
    """
    info_path = f"./datasets/mslr/{task}_{mode}_info.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"Split file not found: {info_path}\n"
            f"Update get_split_order_ids() to point to your actual *_info.json path."
        )

    with open(info_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    # each item has "video_id"
    return [it["video_id"] for it in items]


def write2file(path, info, output):
    """Write token-level CTM. Samples with empty predictions will produce 0 lines."""
    with open(path, "w", encoding="utf-8") as f:
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                f.write(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        info[sample_idx],
                        word_idx / 100.0,
                        (word_idx + 1) / 100.0,
                        word[0],
                    )
                )


def _chosen_ctm_path(work_dir: str, task: str, mode: str):
    # task: "us" -> conv-fusion ; "si" -> fusion
    if task == "us":
        return f"{work_dir}output-hypothesis-conv-fusion-{mode}.ctm"
    else:
        return f"{work_dir}output-hypothesis-fusion-{mode}.ctm"


def read_reference_ids_csv(path: str):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)  # skip header (expects column name: id)
        for row in r:
            if not row:
                continue
            ids.append(str(row[0]).strip())
    return ids

def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, task, evaluate_tool="python"):
    model.eval()
    total_info = []
    total_sent_fusion = []
    total_sent_conv_fusion = []

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        data = device.dict_data_to_device(data)
        with torch.no_grad():
            ret_dict = model(data)

        # origin_info could be like "14_0433|..." or just "08_0001"
        for fn in data["origin_info"]:
            total_info.append(str(fn).split("|")[0])

        total_sent_fusion += ret_dict["recognized_sents_fusion"]
        total_sent_conv_fusion += ret_dict["conv_sents_fusion"]

    python_eval = True if evaluate_tool == "python" else False

    # Write CTMs
    write2file(f"{work_dir}output-hypothesis-fusion-{mode}.ctm", total_info, total_sent_fusion)
    write2file(f"{work_dir}output-hypothesis-conv-fusion-{mode}.ctm", total_info, total_sent_conv_fusion)

    # Choose which CTM to convert based on task
    chosen_ctm = _chosen_ctm_path(work_dir, task, mode)
    word_dict = ctm_to_word_dict(chosen_ctm)

    
    if mode == "test":
    #  use the provided ids-only test.csv as the ordering
        REF_TEST_IDS = "./annotations_v2/isharah2000/SI/test.csv"   # <-- change this path
        ordered_ids = read_reference_ids_csv(REF_TEST_IDS)

        csv_file = f"{work_dir}test.csv"
        write_csv_all_ids(csv_file, ordered_ids, word_dict)

        missing_rows = sum(1 for vid in ordered_ids if vid not in word_dict)
        print(f"[{mode}] empty predictions (no CTM lines): {missing_rows} / {len(ordered_ids)}")
        return csv_file

    # dev: evaluate + also dump csv
    try:
        lstm_ret_fusion = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file=f"output-hypothesis-fusion-{mode}.ctm",
            evaluate_dir=cfg.dataset_info["evaluation_dir"],
            evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
            output_dir=f"epoch_{epoch}_result/",
            python_evaluate=python_eval,
            triplet=True,
        )
        conv_ret_fusion = evaluate(
            prefix=work_dir,
            mode=mode,
            output_file=f"output-hypothesis-conv-fusion-{mode}.ctm",
            evaluate_dir=cfg.dataset_info["evaluation_dir"],
            evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
            output_dir=f"epoch_{epoch}_result/",
            python_evaluate=python_eval,
        )
    except Exception:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret_fusion = 100.0
        conv_ret_fusion = 100.0

    recoder.print_log(
        f"Epoch {epoch}, {mode} Conv1D WER: {conv_ret_fusion: 2.2f}%, BiLSTM WER: {lstm_ret_fusion: 2.2f}%",
        f"{work_dir}/{mode}.txt",
    )

    ordered_ids = get_split_order_ids(cfg, task, mode)
    csv_file = f"{work_dir}{mode}_epoch{epoch}.csv"
    write_csv_all_ids(csv_file, ordered_ids, word_dict)

    missing_rows = sum(1 for vid in ordered_ids if vid not in word_dict)
    print(f"[{mode}] empty predictions (no CTM lines): {missing_rows} / {len(ordered_ids)}")

    return min(conv_ret_fusion, lstm_ret_fusion)
