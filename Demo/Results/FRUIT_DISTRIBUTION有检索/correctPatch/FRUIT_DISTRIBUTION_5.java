public static int fruit_distribution(String s, int n) {
    int result = n;
    String[] fruitCounts = s.split(" ");
    for (String str : fruitCounts) {
        try {
            int cnt = Integer.parseInt(str);
            result -= cnt;
        } catch (NumberFormatException e) {
            // ignore invalid count
        }
    }
    return result;
}