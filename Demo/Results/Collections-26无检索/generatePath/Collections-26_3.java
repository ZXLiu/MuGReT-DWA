private Object readResolve() throws IOException, ClassNotFoundException {
    calculateHashCode(keys);
    return this;
}