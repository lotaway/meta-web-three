public interface IBaseResponse<Data> {
    ResponseStatus status;
    String message;
    Data data;
}
