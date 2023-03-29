package com.metawebthree.server.controller;

import com.metawebthree.server.service.SettingService;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.web.bind.annotation.*;
import com.config.InitScanner;

import java.io.Serial;
import java.io.Serializable;
import java.util.Objects;

class ApiResponse {
    private int status;
    private String message;
    private Object data;

    public ApiResponse() {

    }

    public ApiResponse(int status, String message) {
        this.status = status;
        this.message = message;
    }

    public ApiResponse(int status, String message, Object data) {
        this.status = status;
        this.message = message;
        this.data = data;
    }

    public int getStatus() {
        return status;
    }

    public String getMessage() {
        return message;
    }

    public Object getData() {
        return data;
    }

    public static ApiResponse success() {
        return new ApiResponse(200, "success");
    }

    public static ApiResponse success(Object data) {
        return new ApiResponse(200, "success", data);
    }

    public static ApiResponse error() {
        return new ApiResponse(201, "error");
    }

    public static ApiResponse error(Exception e) {
        return new ApiResponse(201, "error", e);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ApiResponse that = (ApiResponse) o;
        return status == that.status && message.equals(that.message) && data.equals(that.data);
    }

    @Override
    public int hashCode() {
        return Objects.hash(status, message, data);
    }

    @Override
    public String toString() {
        return "ApiResponse{" +
                "status=" + status +
                ", message='" + message + '\'' +
                ", data=" + data +
                '}';
    }
}

@RestController
public class ShowInfo implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;
    @Value("${name}")
    private String name;

    @Autowired
    private Environment env;

    @Autowired
    private SettingService serviceSettingService;

    private final Author author;

    public ShowInfo(Author author) {
        this.author = author;
    }

    @RequestMapping("/name")
    public String name() {
        return name;
    }

    @RequestMapping("/author")
    public String getAuthor() {
        return this.author.toString();
    }

    @RequestMapping("/scanner")
    public String scanner(HttpServletRequest request) throws Exception {
        String showType = request.getParameter("showType");
        return InitScanner.getInfo();
    }

    @RequestMapping(path = "/showConfig", method = RequestMethod.POST)
    public ApiResponse showConfig(@RequestParam String type) {
        return ApiResponse.success("show config");
    }

    @RequestMapping("/showError")
    public String showError(ShowErrorArgs showErrorArgs) throws Exception {
        boolean result = InitScanner.errorOutputToFile("Nothing error");
        return InitScanner.getErrorLog("error/error.log");
    }

    @RequestMapping("/testPath/{id}")
    public String testPath(@PathVariable int id) {
        String fileConfig = serviceSettingService.getTestConfig();
        return "test path parameter id is:" + id + ", and get config file test:" + fileConfig;
    }
}
