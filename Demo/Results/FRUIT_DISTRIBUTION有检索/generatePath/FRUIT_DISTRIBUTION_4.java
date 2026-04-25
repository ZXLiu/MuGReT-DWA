public static int fruit_distribution(String s, int n) {
    int result = n;
    for (String str : s.split(" ")) {
        int cnt = 0;
        try {
            cnt = Integer.parseInt(str);
        } catch (NumberFormatException e) {
            // skip the invalid fruit count
        }
        result -= cnt;
    }
    return result;
}