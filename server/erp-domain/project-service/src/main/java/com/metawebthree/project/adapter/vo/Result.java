package com.metawebthree.project.adapter.vo;

import lombok.Data;
import java.util.List;

@Data
public class Result<T> {
    private Integer code;
    private String message;
    private T data;
    private Long total;

    public static <T> Result<T> success(T data) {
        Result<T> result = new Result<>();
        result.setCode(200);
        result.setMessage("success");
        result.setData(data);
        return result;
    }

    public static <T> Result<PageResult<T>> successPage(List<T> list, Long total) {
        Result<PageResult<T>> result = new Result<>();
        result.setCode(200);
        result.setMessage("success");
        PageResult<T> pageResult = new PageResult<>();
        pageResult.setList(list);
        pageResult.setTotal(total);
        result.setData(pageResult);
        return result;
    }

    public static <T> Result<T> fail(String message) {
        Result<T> result = new Result<>();
        result.setCode(500);
        result.setMessage(message);
        return result;
    }

    @Data
    public static class PageResult<T> {
        private List<T> list;
        private Long total;
    }
}