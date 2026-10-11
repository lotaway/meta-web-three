package com.metawebthree.hrm.domain.exception;

public class DepartmentNotFoundException extends RuntimeException {
    public DepartmentNotFoundException(Long id) {
        super("Department not found: " + id);
    }
}