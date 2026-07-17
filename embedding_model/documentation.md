Trains embedding model to reproduce the LLM-judges

Data input:
    consciousness_filter/data/fineweb_edu_88k_rated.jsonl
    consciousness_filter/data/hand_annotated_samples.jsonl

    Creates:
        validation set = hand_annotated samples + uniformly random 5% random sample of fineweb_edu_88k_rated
        train set = fineweb_edu_88k_rated - validation set

        Note: the validation set contains the hand_annotated samples, but uses the LLM judges ratings on those samples

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


    





