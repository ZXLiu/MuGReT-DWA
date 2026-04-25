public static int fruit_distribution(String s, int n) {
    int result = n;
    String[] strArray = s.split(" ");
    for (int i = 0; i < strArray.length; i++) {
        try {
            int cnt = Integer.parseInt(strArray[i]);
            result -= cnt;
        } catch (Exception e) {
            // skip the invalid count
        }
    }
    return result;
}


// Provide a fix for the buggy