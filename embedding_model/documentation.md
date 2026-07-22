Trains embedding model to reproduce the LLM-judges

Data input:
    data.rated_path             the LLM-rated corpus (fineweb_edu_88k_rated.jsonl)
    data.hand_rated_path        llm_judge/data/hand_annotated_rated.jsonl
                                    written by llm_judge/scripts/rate_hand_filter.py
                                    one row per hand-annotated document, carrying both
                                    the human labels and the LLM judge ratings
    data.hand_annotations_path  llm_judge/data/hand_annotated_samples.jsonl
                                    source of truth for the human labels, so relabeling
                                    takes effect without re-running the hand rater

    Creates:
        validation set = every hand_rated row with complete judge targets
                         + the configured uniformly random fraction of the remaining rated corpus
        train set = rated corpus - hand documents - random validation rows

        Note: the validation set contains the hand_annotated samples, but uses the LLM judges ratings on those samples

    Disjointness:
        The hand documents live in their own file now, so a copy of one may also sit in the
        rated corpus. Any corpus row matching a hand document - exactly, or on the first
        data.prefix_match_chars characters - is dropped from training. The exclusion uses
        every hand annotation, including ones the hand rater has not rated yet, so a
        partially rated hand file can never leak into training.

    Training-only upsampling:
        data.upsample_mult defines an integer weight for every filter
        A training row qualifies for a filter when its mean configured-judge rating is >= 2
        Weight 1 keeps one copy; weight N gives the row N total copies
        If a row qualifies for multiple filters, only the largest qualifying weight is used
        Validation rows are never duplicated

Model:
    Modern Bert, but we append a prediction head at the end, which has (n_judges in fineweb_edu_rated x n_filters) outputs
    This is just a matrix multiply from d_model -> n_judges x n_filters


Optimization:
    Use adamw (betas and learning rate in config)
    Use lora (rank in train config) on embedding, attention matrices and mlp matrices
        - Do not train biases and norm parameters
    Prediction head is initialized at zero, trained fully (no lora) from scratch

    Set grad clip = 1.0
    Set wamup to 10% of training
    Use cosine annealing decay for last 30% of training
    Stable at peak lr for the 60% rest


Pipeline:
    Take train and validation, tokenize them

    Compute (n_judges x n_filters) labels for each document by mapping judge ratings to [0,1] for each filter rating, by dividing by 10 (so 0 -> 0%, 1 -> 10%, 2 -> 20%)

    shuffle train and cut it into batch_size batches

    For each batch:
        Pass the tokenized batches through bert
            - Pad them to max sample length
        gather logits from the prediction head at token position zero
        
        loss = binary_cross_entropy_with_logits(logits,ratings / 10,reduction="none")

        take mean over the batch dimension and the n_judges x n_filters dimension
        backpropagate
        do an optimizer step


Logging:
    every log_every (in config), print loss and grad norm to terminal
    every val_every print validation loss (same objective as train, but evaluated on validation)
        - run over whole validation set, with same batch size as train


    



