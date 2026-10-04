import math, re

dataset = [
    ("Hey, are we meeting for lunch today?", "Ham"),
    ("Congratulations! You have won a free ticket to the concert.", "Spam"),
    ("Please send me the assignment before tomorrow.", "Ham"),
    ("URGENT! Claim your cash prize now.", "Spam"),
    ("Are you coming to the college today?", "Ham"),
    ("Win a free iPhone by clicking this link.", "Spam"),
    ("I will call you after the meeting.", "Ham"),
    ("You have been selected for a lottery worth 50000 rupees.", "Spam"),
    ("Please bring your notebook to class.", "Ham"),
    ("Get a special discount! Buy now and save 70 percent.", "Spam"),
    ("Happy birthday! Have a wonderful day.", "Ham"),
    ("Claim your free gift card. Limited offer!", "Spam"),
    ("The lecture will start at 10 AM tomorrow.", "Ham"),
    ("You won a cash reward. Call now to claim your prize.", "Spam"),
    ("Can you share your machine learning notes?", "Ham"),
    ("Exclusive offer! Get a free vacation package now.", "Spam"),
    ("I have reached the library. Where are you?", "Ham"),
    ("Congratulations! You are our lucky winner.", "Spam"),
    ("Let's meet at the cafeteria after class.", "Ham"),
    ("Urgent! You have won a free shopping voucher.", "Spam")
]

def tokenize(text):
    return re.findall(r'\w+', text.lower())

# Counts and Vocab
ham_words, spam_words = {}, {}
n_ham, n_spam = 0, 0
vocab = set()

for msg, label in dataset:
    words = tokenize(msg)
    vocab.update(words)
    target_dict = spam_words if label == "Spam" else ham_words
    for w in words:
        target_dict[w] = target_dict.get(w, 0) + 1
    if label == "Spam": n_spam += len(words)
    else: n_ham += len(words)

# Prediction Logic
def predict(msg):
    words = tokenize(msg)
    s_ham, s_spam = math.log(0.5), math.log(0.5)
    for w in words:
        s_ham += math.log((ham_words.get(w, 0) + 1) / (n_ham + len(vocab)))
        s_spam += math.log((spam_words.get(w, 0) + 1) / (n_spam + len(vocab)))
    return "Spam" if s_spam > s_ham else "Ham"

# Testing
test_messages = [
    "Congratulations! You have won a free ticket to the concert.",
    "Hey, are we meeting for lunch today?",
    "URGENT! Claim your cash prize now.",
    "Please send me the assignment before tomorrow."
]

for msg in test_messages:
    print(f'"{msg}" -> {predict(msg)}')