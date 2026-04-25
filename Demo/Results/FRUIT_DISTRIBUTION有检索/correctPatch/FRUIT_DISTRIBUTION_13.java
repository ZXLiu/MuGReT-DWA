public static int fruit_distribution(String s, int n) {
    int result = n;
    for (String str : s.split(" ")) {
        try {
            int cnt = Integer.parseInt(str);
            result -= cnt;
        } catch (NumberFormatException e) {
            // skip the invalid number
        }
    }
    return result;
}