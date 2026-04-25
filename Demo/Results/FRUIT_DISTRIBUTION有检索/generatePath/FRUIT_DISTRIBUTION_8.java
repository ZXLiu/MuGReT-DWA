public static int fruit_distribution(String s, int n) {
    int result = n;
    for (String str : s.split(" ")) {
        int cnt;
        try {
            cnt = Integer.parseInt(str);
        } catch (Exception e) {
            return -1;
        }
        result -= cnt;
    }
    return result;
}