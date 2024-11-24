from trl import SFTTrainer


class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, data_collator, tokenizer, compute_metrics, preprocess_logits_for_metrics, sft_config, peft_config, optimizers=None):
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            args=sft_config,
            peft_config=peft_config,
            optimizers=optimizers,
        )

    def train(self):
        self.trainer.train()
