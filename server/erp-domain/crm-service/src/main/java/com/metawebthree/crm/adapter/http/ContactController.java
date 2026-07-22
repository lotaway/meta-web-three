package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.client.UserServiceClient;
import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.application.command.ContactCommandService;
import com.metawebthree.crm.application.query.ContactQueryService;
import com.metawebthree.crm.domain.entity.Contact;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/crm/contacts")
@RequiredArgsConstructor
public class ContactController {

    private final ContactQueryService contactQueryService;
    private final ContactCommandService contactCommandService;
    private final UserServiceClient userServiceClient;

    @GetMapping("/{id}")
    public Result<Contact> getById(@PathVariable Long id) {
        return Result.success(contactQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<Contact>> listAll() {
        return Result.success(contactQueryService.listAll());
    }

    @GetMapping("/customer/{customerId}")
    public Result<List<Contact>> getByCustomer(@PathVariable Long customerId) {
        return Result.success(contactQueryService.listByCustomerId(customerId));
    }

    @PostMapping
    public Result<Contact> create(@RequestBody Contact contact) {
        return Result.success(contactCommandService.create(contact));
    }

    @PutMapping
    public Result<Contact> update(@RequestBody Contact contact) {
        return Result.success(contactCommandService.update(contact));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        contactCommandService.delete(id);
        return Result.success();
    }

    @GetMapping("/sync/customer/{userId}")
    public Result<UserServiceClient.UserDTO> syncCustomerData(@PathVariable Long userId) {
        UserServiceClient.UserDTO user = userServiceClient.getUserById(userId);
        if (user == null) {
            return Result.error("Customer not found in user-service");
        }
        return Result.success(user);
    }
}
