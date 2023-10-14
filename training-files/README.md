# Training Files

To train both the Text-to-Template and Template-Fill-In model, we used [PICARD](https://github.com/ServiceNow/picard) to train our FLAN-T5-XL models. The datasets we use are provided in the JSON format for the two tasks under `text-to-template/` and `template-fill-in/`, respectively.

Their code can be used as-is with the following modifications. 

First, you need to modify the input format in the `spider_pre_process_function()` function found under `seq2seq/spider/utils.py` in the PICARD repository. We modified it as follows, to work with our JSON files instead of the default SPIDER splits:

```python
def spider_get_input_based_on_type(
    question: str,
    serialized_schema: str,
    task_type: str = "semantic parse",
    template = None
) -> str:
    prefix = task_type + ": "
    if task_type in ["semantic parse", "template generation"]:
        ip = prefix + question.strip() + " " + serialized_schema.strip()
    else:
        # template fill
        ip = prefix + question.strip() + " " + serialized_schema.strip() + \
            " @ " + template.strip() 
    return ip

def spider_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""
    if "task_type" in batch:
        inputs = [
            spider_get_input_based_on_type(question=question, \
                serialized_schema=serialized_schema, task_type=task_type, \
                template=template) for question, serialized_schema, task_type, \
                template in zip(batch["question"], \
                batch["schema"] if "schema" in batch else \
                batch["serialized_schema"], batch["task_type"], batch["template"])
        ]
    else:
        inputs = [
            spider_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
            for question, serialized_schema in zip(batch["question"], \
                batch["serialized_schema"])
        ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [
        spider_get_target(
            query=query,
            db_id=db_id,
            normalize_query=data_training_args.normalize_query,
            target_with_db_id=data_training_args.target_with_db_id,
        )
        for db_id, query in zip(batch["db_id"], batch["query"])
    ]

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

In addition, you will also have to update the `metas` variable in the `_post_process_function` of `class SpiderTrainer` in `seq2seq/utils/spider.py`:
```python
    ...
    "db_foreign_keys": x["db_foreign_keys"],
    "query_type": "sql" if "query_type" not in x else x["query_type"]
}
``` 

The input format for Text-to-Template generation is
```
template generation: <question> | <db_id> | <schema_without_content>
```
and the output format is (the templates in the JSONs already have the JOINs/SELECTs prefix in them)
```
<db_id> | <J> joins @ <S> selects @ <template>
```
We use an exact string match of templates after case/space normalization as our metric to choose the best model. For Template Generation, we follow the original choice of PICARD in using the Exact Match metric. There, the input format is
```
template fill: <question> | <db_id> | <schema_without_content> @ <template>
```

The metrics go under `seq2seq/metrics/spider/` in the PICARD repository. We show an example metric in `metrics/spider_combined_match.py` and the corresponding modification to `seq2seq/metrics/spider/spider.py` in `metrics/spider.py`.

We use all the default PICARD parameters.
