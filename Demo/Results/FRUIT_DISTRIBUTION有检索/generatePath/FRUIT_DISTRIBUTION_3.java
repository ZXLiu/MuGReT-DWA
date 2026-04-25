public static int fruit_distribution(String s, int n) {
    int result = n;
    for (String str : s.split(" ")) {
        int cnt = Integer.parseInt(str);
        if (cnt > n) {
            return -1;
        }
        result -= cnt;
    }
    return result;
}