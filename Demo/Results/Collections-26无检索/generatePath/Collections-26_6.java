private Object readResolve() {
    calculateHashCode(keys);
    return new Data(keys, values);
}