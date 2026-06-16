def get_length(arr):
    count = 0
    for _ in arr:
        count += 1
    return count

def get_max(arr):
    if not arr:
        return 0
    maximum = arr[0]
    for item in arr:
        if item > maximum:
            maximum = item
    return maximum

def counting_sort_for_radix(arr, exp):
    n = get_length(arr)
    
    output = []
    i = 0
    while i < n:
        output.append(0)
        i += 1
        
    count = []
    i = 0
    while i < 10:
        count.append(0)
        i += 1
    
    i = 0
    while i < n:
        index = (arr[i] // exp) % 10
        count[index] += 1
        i += 1
        
    i = 1
    while i < 10:
        count[i] += count[i - 1]
        i += 1
        
    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1
        
    i = 0
    while i < n:
        arr[i] = output[i]
        i += 1

def radix_sort(arr):
    n = get_length(arr)
    if n == 0:
        return arr
        
    max_val = get_max(arr)
    exp = 1
    while max_val // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
        
    return arr

if __name__ == "__main__":
    data_radix = [170, 45, 75, 90, 802, 24, 2, 66]
    print(radix_sort(data_radix))
