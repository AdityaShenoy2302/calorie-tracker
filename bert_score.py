# BERTScore:
# Instead of checking exact words, BERTScore uses a language model (like BERT) to check meaning similarity.
# Even if your candidate uses different numbers/words but keeps the meaning, BERTScore will give high score.
# Much better for chatbots, QA systems, and real-world responses.

from bert_score import score

# Your sentences
reference = ["You likely consumed around 400–700 calories, maybe more depending on what else was inside and how big it was."]
candidate = ["According to the provided context, a Burrito with cheese only has approximately 328 calories."]

# Compute BERTScore
P, R, F1 = score(candidate, reference, lang="en", verbose=True)

# Show results
print(f"BERTScore Precision: {P.mean():.4f}")
print(f"BERTScore Recall: {R.mean():.4f}")
print(f"BERTScore F1: {F1.mean():.4f}")
