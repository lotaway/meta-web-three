package com.metawebthree.user;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/userType")
class UserRoleController {

    private final UserRoleService userRoleService;

    UserRoleController(UserRoleService userRoleService) {
        this.userRoleService = userRoleService;
    }

    @GetMapping("/list")
    String getRegisterType() {
        return userRoleService.getList().toString();
    }
}