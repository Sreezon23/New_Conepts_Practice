from itertools import combinations


def get_frequent_itemsets(transactions, min_support):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[frozenset([item])] = item_counts.get(frozenset([item]), 0) + 1

    transactions_count = len(transactions)
    frequent_itemsets = {
        item: count
        for item, count in item_counts.items()
        if count / transactions_count >= min_support
    }

    k = 2
    current_itemsets = list(frequent_itemsets.keys())

    while current_itemsets:
        candidates = set()
        for i in range(len(current_itemsets)):
            for j in range(i + 1, len(current_itemsets)):
                candidate = current_itemsets[i] | current_itemsets[j]
                if len(candidate) == k:
                    candidates.add(candidate)

        candidate_counts = {candidate: 0 for candidate in candidates}
        for transaction in transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidate_counts[candidate] += 1

        current_itemsets = [
            itemset
            for itemset, count in candidate_counts.items()
            if count / transactions_count >= min_support
        ]
        for itemset in current_itemsets:
            frequent_itemsets[itemset] = candidate_counts[itemset]

        k += 1

    return frequent_itemsets, transactions_count


def generate_association_rules(frequent_itemsets, transactions_count, min_confidence):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                antecedent_support = frequent_itemsets.get(antecedent)
                if antecedent_support is None:
                    continue

                confidence = frequent_itemsets[itemset] / antecedent_support
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, confidence))

    return sorted(rules, key=lambda x: (-x[2], len(x[0]), len(x[1])))


def print_frequent_itemsets(frequent_itemsets, transactions_count):
    print("Frequent itemsets:")
    for itemset, count in sorted(frequent_itemsets.items(), key=lambda x: (len(x[0]), x[0])):
        print(f"  {set(itemset)} - support: {count}/{transactions_count} = {count / transactions_count:.2f}")
    print()


def print_association_rules(rules):
    print("Association rules:")
    for antecedent, consequent, confidence in rules:
        print(
            f"  {set(antecedent)} => {set(consequent)} "
            f"(confidence: {confidence:.2f})"
        )
    print()


def main():
    transactions = [
        ["milk", "bread", "eggs"],
        ["milk", "bread"],
        ["milk", "diapers", "beer", "bread"],
        ["bread", "eggs", "diapers"],
        ["milk", "bread", "eggs", "diapers"],
        ["milk", "bread", "beer"],
    ]

    min_support = 0.5
    min_confidence = 0.7

    frequent_itemsets, transactions_count = get_frequent_itemsets(transactions, min_support)
    print_frequent_itemsets(frequent_itemsets, transactions_count)

    rules = generate_association_rules(frequent_itemsets, transactions_count, min_confidence)
    print_association_rules(rules)


if __name__ == "__main__":
    main()
