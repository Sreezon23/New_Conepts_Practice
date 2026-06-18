def get_length(arr):
    count = 0
    for _ in arr:
        count += 1
    return count

def simple_sort(arr):
    n = get_length(arr)
    i = 0
    while i < n:
        j = 0
        while j < n - i - 1:
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp
            j += 1
        i += 1
    return arr

def remove_duplicates(arr):
    unique = []
    for item in arr:
        found = False
        for u in unique:
            if u == item:
                found = True
                break
        if not found:
            unique.append(item)
    return unique

def compress_array(arr):
    unique_items = remove_duplicates(arr)
    sorted_unique = simple_sort(unique_items)
    
    value_to_index = {}
    idx = 0
    for val in sorted_unique:
        value_to_index[val] = idx
        idx += 1
        
    compressed = []
    for val in arr:
        compressed.append(value_to_index[val])
        
    return compressed

if __name__ == "__main__":
    arr = [1000, 20, 1000, 300, 20, 50000]
    print(compress_array(arr))
