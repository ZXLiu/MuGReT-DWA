public static int fruit_distribution(String s, int n) {
    int result = n;
    String[] strings = s.split(" ");
    for (String str : strings) {
        int cnt = Integer.parseInt(str);
        result -= cnt;
    }
    return result;
}