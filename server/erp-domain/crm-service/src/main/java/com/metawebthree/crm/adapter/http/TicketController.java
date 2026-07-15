package com.metawebthree.crm.adapter.http;

import com.metawebthree.crm.adapter.vo.Result;
import com.metawebthree.crm.application.command.TicketCommandService;
import com.metawebthree.crm.application.query.TicketQueryService;
import com.metawebthree.crm.domain.entity.CustomerServiceTicket;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/crm/tickets")
@RequiredArgsConstructor
public class TicketController {

    private final TicketQueryService ticketQueryService;
    private final TicketCommandService ticketCommandService;

    @GetMapping("/{id}")
    public Result<CustomerServiceTicket> getById(@PathVariable Long id) {
        return Result.success(ticketQueryService.getById(id));
    }

    @GetMapping("/list")
    public Result<List<CustomerServiceTicket>> listAll() {
        return Result.success(ticketQueryService.listAll());
    }

    @GetMapping("/status/{status}")
    public Result<List<CustomerServiceTicket>> listByStatus(@PathVariable String status) {
        return Result.success(ticketQueryService.listByStatus(status));
    }

    @GetMapping("/priority/{priority}")
    public Result<List<CustomerServiceTicket>> listByPriority(@PathVariable String priority) {
        return Result.success(ticketQueryService.listByPriority(priority));
    }

    @GetMapping("/type/{type}")
    public Result<List<CustomerServiceTicket>> listByType(@PathVariable String type) {
        return Result.success(ticketQueryService.listByType(type));
    }

    @GetMapping("/assigned/{assignedTo}")
    public Result<List<CustomerServiceTicket>> listByAssignedTo(@PathVariable String assignedTo) {
        return Result.success(ticketQueryService.listByAssignedTo(assignedTo));
    }

    @GetMapping("/customer/{customerId}")
    public Result<List<CustomerServiceTicket>> listByCustomerId(@PathVariable Long customerId) {
        return Result.success(ticketQueryService.listByCustomerId(customerId));
    }

    @GetMapping("/search")
    public Result<List<CustomerServiceTicket>> search(@RequestParam String keywords) {
        return Result.success(ticketQueryService.search(keywords));
    }

    @PostMapping
    public Result<CustomerServiceTicket> create(@RequestBody CustomerServiceTicket ticket) {
        return Result.success(ticketCommandService.create(ticket));
    }

    @PutMapping("/{id}/assign")
    public Result<CustomerServiceTicket> assign(@PathVariable Long id, @RequestParam String assignedTo) {
        return Result.success(ticketCommandService.assign(id, assignedTo));
    }

    @PutMapping("/{id}/status")
    public Result<CustomerServiceTicket> updateStatus(@PathVariable Long id, @RequestParam String status) {
        return Result.success(ticketCommandService.updateStatus(id, status));
    }

    @DeleteMapping("/{id}")
    public Result<Void> delete(@PathVariable Long id) {
        ticketCommandService.delete(id);
        return Result.success();
    }
}
