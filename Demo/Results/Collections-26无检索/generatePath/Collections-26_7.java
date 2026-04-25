private Object readResolve() {
    calculateHashCode(keys);
    return readObject();
}