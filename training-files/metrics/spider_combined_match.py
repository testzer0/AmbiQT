"""Spider combined match metric."""

from typing import Dict, Any
from third_party.spider import evaluation as spider_evaluation
from seq2seq.metrics.spider.spider_exact_match import compute_exact_match_metric
from seq2seq.metrics.spider.spider_test_suite import compute_test_suite_metric

tables = None
tab_set_lower = None
col_set_lower = None

def templatize_sql(sql, db_id):
    global tables, tab_set_lower, col_set_lower
    sql = sql.replace("!=", " !=")
    sql = " ".join(sql.split())
    if tables is None:
        tables = json.load(open("raid/infolab/adithyabhaskar/text2sql/datasets/"+\
            "Spider/spider/tables.json"))
        tables = {x['db_id']: x for x in tables}
    if db_id not in tab_set_lower:
        tab_set_lower[db_id] = set(x.lower() for x in \
            tables[db_id]["table_names_original"])
        col_set_lower[db_id] = set(x[1].lower() for x in \
            tables[db_id]["column_names_original"])
    if sql[-1] == ';':
        sql = sql[:-1]
    sql = sql.replace("''", "\"").replace(",", " , ").\
        replace("(", " ( ").replace(")", " ) ").split()
    template = []
    current_quote = None
    for token in sql:
        if current_quote is not None:
            if token[-1] == current_quote:
                current_quote = None
                template.append("string")
        else:
            if token[0] in ['\'', '"', '`']:
                if len(token) > 1 and token[-1] == token[0]:
                    template.append("string")
                else:
                    current_quote = token[0]
            elif token.isnumeric():
                template.append("number")
            else:
                if len(token) > 3 and token[0] in ('t', 'T') and token[2] == '.':
                    token_pref = token[3:].lower()
                else:
                    token_pref = token.lower()
                if token_pref in tab_set_lower[db_id]:
                    template.append("table")
                elif token_pref in col_set_lower[db_id]:
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    template = template.lower()
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return template

def normalize_template_for_comparison(template):
    template = template.lower().replace("!=", " !=")
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return template

def compute_template_match_metric(predictions, references) -> Dict[str, Any]:
    n = 0
    score = 0
    for prediction, reference in zip(predictions, references):
        template = reference["query"]
        prediction = normalize_template_for_comparison(prediction)
        template = normalize_template_for_comparison(template)
        n += 1
        score += int(template == prediction)
    score /= n
    return {
        "template_match": score,
    }
    
def compute_combined_match_metric(predictions, references) -> Dict[str, Any]:
    predictions_for_exact_match = []
    references_for_exact_match = []
    predictions_for_template_match = []
    references_for_template_match = []
    
    for prediction, reference in zip(predictions, references):
        if reference["query_type"] == "template":
            predictions_for_template_match.append(prediction)
            references_for_template_match.append(reference)
        else:
            predictions_for_exact_match.append(prediction)
            references_for_exact_match.append(reference)
    
    em_metric = compute_exact_match_metric(predictions_for_exact_match, \
        references_for_exact_match)["exact_match"] 
    tm_metric = compute_template_match_metric(predictions_for_template_match, \
        references_for_template_match)["template_match"]
    combined_metric = (em_metric*len(predictions_for_exact_match) + \
        tm_metric*len(predictions_for_template_match)) / \
        (len(predictions_for_exact_match) + len(predictions_for_template_match))
    
    ts_metric = compute_test_suite_metric(predictions_for_exact_match, \
        references_for_exact_match)["exec"]
    
    return {
        "combined_match": combined_metric,
        "exact_match": em_metric,
        "template_match": tm_metric,
        "eval_match": ts_metric
    }