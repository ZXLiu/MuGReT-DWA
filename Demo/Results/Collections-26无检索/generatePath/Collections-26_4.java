private Object readResolve() {
    calculateHashCode(keys);
    return new HashMap(keys, values);
}