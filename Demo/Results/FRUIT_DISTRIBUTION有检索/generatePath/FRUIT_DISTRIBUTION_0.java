public static int fruit_distribution(String s, int n) {
    int result = n;
    String[] fruits = s.split(" ");
    for (String str : fruits) {
        int cnt = Integer.parseInt(str);
        result -= cnt;
    }
    return result;
}