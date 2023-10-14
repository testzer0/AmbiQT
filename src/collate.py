import sys
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-path", "-i", help="Path to file with predictions", default=None)
    parser.add_argument("--t5-3b-path", "-t", help="File with predictions of baseline T5-3B", \
        default=None)
    parser.add_argument("--out-path", "-o", help="Path for output file")
    parser.add_argument("--k", "-k", type=int, default=5, help="Number of outputs per instance")
    parser.add_argument("--num-from-t5", "-n", type=int, default=2, \
        help="How many outputs to include from T5")
    args = parser.parse_args()
    
    assert args.in_path is not None, "Need an input file!"
    assert args.t5_3b_path is not None, "Need T5-3B (vanilla)'s predictions to collate!"
    if args.out_path is None:
        args.out_path = args.in_path[:args.in_path.rfind('.json')] + "-collated.json"
    return args

def get_selections(selections, k=10):
    # Note that its OK to select many examples here, as we will truncate after deduplication
    if type(selections[0]) == str:
        return selections
    sel_with_scores = []
    for i in range(len(selections)):
        for j in range(len(selections[i])):
            sel_with_scores.append((selections[i][j], 3*i+2*j))
    sel_with_scores = sorted(sel_with_scores, key=lambda x: x[1])
    final = []
    for sel, _ in sel_with_scores:
        if len(final) >= k:
            break
        if sel not in final:
            final.append(sel)
    return final

def lower(s):
    s = s.replace("``", "`")
    lowers = ""
    current_quote = None
    for c in s:
        if current_quote is None:
            lowers += c.lower()
            if c in ['"', '\'', '`']:
                current_quote = c
        else:
            lowers += c
            if c == current_quote:
                current_quote = None
    return lowers

def normalize_sql(sql):
    if sql[-1] == ';':
        sql = sql[:-1]
    sql = " ".join(lower(sql).strip().split())
    sql = sql.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        sql = sql.replace(kword+" (", kword+"(")
    return sql.replace(") ,", "),")

def deduplicate(selections):
    deduped = []
    for selection in selections:
        selection = normalize_sql(selection)
        if selection not in deduped:
            deduped.append(selection)
    return deduped

def collate(in_path, collate_path, out_path, k, n):
    data = json.load(open(in_path))
    ref = json.load(open(collate_path))
    key = 't2s_outs' if 't2s_outs' in data[0] else ('template_filled' if \
        'template_filled' in data[0] else ('chatgpt_out' if 'chatgpt_out' \
        in data[0] else 'flan_direct_outs'))
    for i in range(len(data)):
        data[i][key] = deduplicate(ref[i]['t2s_outs'][:n] + get_selections(data[i][key]))[:k]
    json.dump(data, open(out_path, 'w+'), indent=4)

if __name__ == '__main__':
    args = parse_args()
    collate(args.in_path, args.t5_3b_path, args.out_path, args.k, args.num_from_t5)