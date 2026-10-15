package com.metawebthree.mes.adapter.http.controller;

import org.springframework.core.annotation.Order;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.server.ResponseStatusException;
import java.util.Map;

@Order(-10)
@RestControllerAdvice(basePackages = "com.metawebthree.mes.adapter.http.controller")
public class MesOperationExceptionHandler {
    @ExceptionHandler(com.metawebthree.mes.domain.exception.EquipmentException.class)
    public ResponseEntity<?> equipment(com.metawebthree.mes.domain.exception.EquipmentException error) {
        return response(409, error.getMessage());
    }
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<?> invalid(IllegalArgumentException error) { return response(400, error.getMessage()); }
    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<?> conflict(IllegalStateException error) { return response(409, error.getMessage()); }
    @ExceptionHandler(UnsupportedOperationException.class)
    public ResponseEntity<?> unsupported(UnsupportedOperationException error) { return response(422, error.getMessage()); }
    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<?> integrity(DataIntegrityViolationException error) { return response(409, "Duplicate code or invalid referenced configuration"); }
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<?> validation(MethodArgumentNotValidException error) { return response(400, "Invalid request fields"); }
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<?> status(ResponseStatusException error) { return response(error.getStatusCode().value(), "Resource not found or operation rejected"); }
    private ResponseEntity<?> response(int status, String message) {
        return ResponseEntity.status(status).body(Map.of("success", false, "message", message == null ? "Operation rejected" : message));
    }
}
