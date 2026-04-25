private Object readResolve() throws ObjectStreamException {
    calculateHashCode(keys);
    return this;
}