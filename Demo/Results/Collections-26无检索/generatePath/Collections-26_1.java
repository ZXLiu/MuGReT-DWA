private Object readResolve() {
    calculateHashCode(keys);
    return readResolve();
}