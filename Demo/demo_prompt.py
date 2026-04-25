PROMPT_NO_RAG = """// Provide a fix for the buggy function

// Buggy Function
public static int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r + l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Fixed Function
public static int binarySearch(int arr[], int l, int r, int x)
{{
    if (r >= l) {{
        int mid = l + (r - l) / 2;
        if (arr[mid] == x)
            return mid;
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);
        return binarySearch(arr, mid + 1, r, x);
    }}
    return -1;
}}

// Provide a fix for the buggy function

// Buggy Function
{buggy_code}

// Fixed Function
"""

PROMPT_RAG = """// Provide a fix for the buggy function

// Buggy Function
{retrieval_buggy_code}

// Fixed Function
{retrieval_fixed_code}

// Provide a fix for the buggy function

// Buggy Function
{buggy_code}

// Fixed Function
"""
