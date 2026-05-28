package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.application.query.MesQueryService;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.exception.ProcessRouteException;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.interfaces.dto.ProcessRouteDTO;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/process-route")
@RequiredArgsConstructor
public class ProcessRouteController {
    
    private final ProcessRouteRepository processRouteRepository;
    private final MesQueryService queryService;
    
    @PostMapping
    public ResponseEntity<ProcessRouteDTO> create(@RequestBody ProcessRouteDTO.CreateRequest request) {
        ProcessRoute route = new ProcessRoute();
        route.create(request.getRouteCode(), request.getRouteName(), request.getProductCode());
        
        if (request.getSteps() != null) {
            route.setSteps(request.getSteps().stream()
                .map(stepDto -> stepDto.toEntity())
                .collect(Collectors.toList()));
        }
        
        ProcessRoute.ValidationResult validationResult = route.validateSequence();
        if (!validationResult.isValid()) {
            ProcessRouteDTO dto = new ProcessRouteDTO();
            dto.setValidationResult(false);
            dto.setValidationMessage(String.join("; ", validationResult.getErrors()));
            return ResponseEntity.badRequest().body(dto);
        }
        
        ProcessRoute saved = processRouteRepository.save(route);
        return ResponseEntity.ok(ProcessRouteDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    public ResponseEntity<ProcessRouteDTO> update(
            @PathVariable Long id,
            @RequestBody ProcessRouteDTO.UpdateRequest request) {
        
        ProcessRoute route = processRouteRepository.findById(id)
            .orElseThrow(() -> ProcessRouteException.notFound(id));
        
        route.setRouteName(request.getRouteName());
        route.setProductCode(request.getProductCode());
        
        if (request.getSteps() != null) {
            route.setSteps(request.getSteps().stream()
                .map(stepDto -> stepDto.toEntity())
                .collect(Collectors.toList()));
        }
        
        ProcessRoute.ValidationResult validationResult = route.validateSequence();
        if (!validationResult.isValid()) {
            ProcessRouteDTO dto = ProcessRouteDTO.fromEntity(route);
            dto.setValidationResult(false);
            dto.setValidationMessage(String.join("; ", validationResult.getErrors()));
            return ResponseEntity.badRequest().body(dto);
        }
        
        processRouteRepository.update(route);
        return ResponseEntity.ok(ProcessRouteDTO.fromEntity(route));
    }
    
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        processRouteRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<ProcessRouteDTO> getById(@PathVariable Long id) {
        return processRouteRepository.findById(id)
            .map(route -> ResponseEntity.ok(ProcessRouteDTO.fromEntity(route)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{routeCode}")
    public ResponseEntity<ProcessRouteDTO> getByCode(@PathVariable String routeCode) {
        return processRouteRepository.findByRouteCode(routeCode)
            .map(route -> ResponseEntity.ok(ProcessRouteDTO.fromEntity(route)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/product/{productCode}")
    public ResponseEntity<List<ProcessRouteDTO>> getByProduct(@PathVariable String productCode) {
        List<ProcessRouteDTO> routes = processRouteRepository.findByProductCode(productCode)
            .stream()
            .map(ProcessRouteDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(routes);
    }
    
    @GetMapping
    public ResponseEntity<List<ProcessRouteDTO>> list(
            @RequestParam(required = false) String status) {
        
        List<ProcessRoute> routes;
        if (status != null && !status.isEmpty()) {
            ProcessRoute.RouteStatus routeStatus = ProcessRoute.RouteStatus.valueOf(status);
            routes = processRouteRepository.findByStatus(routeStatus);
        } else {
            List<ProcessRoute> allRoutes = new ArrayList<>();
            allRoutes.addAll(processRouteRepository.findByStatus(ProcessRoute.RouteStatus.DRAFT));
            allRoutes.addAll(processRouteRepository.findByStatus(ProcessRoute.RouteStatus.ACTIVE));
            allRoutes.addAll(processRouteRepository.findByStatus(ProcessRoute.RouteStatus.ARCHIVED));
            routes = allRoutes;
        }
        
        List<ProcessRouteDTO> dtos = routes.stream()
            .map(ProcessRouteDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping("/{id}/activate")
    public ResponseEntity<ProcessRouteDTO> activate(@PathVariable Long id) {
        ProcessRoute route = processRouteRepository.findById(id)
            .orElseThrow(() -> ProcessRouteException.notFound(id));
        
        ProcessRoute.ValidationResult validationResult = route.validateSequence();
        if (!validationResult.isValid()) {
            ProcessRouteDTO dto = ProcessRouteDTO.fromEntity(route);
            dto.setValidationResult(false);
            dto.setValidationMessage("激活失败: " + String.join("; ", validationResult.getErrors()));
            return ResponseEntity.badRequest().body(dto);
        }
        
        route.activate();
        processRouteRepository.update(route);
        return ResponseEntity.ok(ProcessRouteDTO.fromEntity(route));
    }
    
    @PostMapping("/{id}/archive")
    public ResponseEntity<ProcessRouteDTO> archive(@PathVariable Long id) {
        ProcessRoute route = processRouteRepository.findById(id)
            .orElseThrow(() -> ProcessRouteException.notFound(id));
        
        route.archive();
        processRouteRepository.update(route);
        return ResponseEntity.ok(ProcessRouteDTO.fromEntity(route));
    }
    
    @PostMapping("/{id}/validate")
    public ResponseEntity<ProcessRouteDTO> validate(@PathVariable Long id) {
        ProcessRoute route = processRouteRepository.findById(id)
            .orElseThrow(() -> ProcessRouteException.notFound(id));
        
        ProcessRoute.ValidationResult validationResult = route.validateSequence();
        ProcessRouteDTO dto = ProcessRouteDTO.fromEntity(route);
        dto.setValidationResult(validationResult.isValid());
        dto.setValidationMessage(validationResult.isValid() ? "验证通过" : String.join("; ", validationResult.getErrors()));
        
        return ResponseEntity.ok(dto);
    }
    
    @GetMapping("/{id}/next-step/{stepNo}")
    public ResponseEntity<ProcessRouteDTO.ProcessStepDTO> getNextStep(
            @PathVariable Long id,
            @PathVariable Integer stepNo) {
        
        return processRouteRepository.findById(id)
            .flatMap(route -> route.getNextStep(stepNo))
            .map(step -> ResponseEntity.ok(ProcessRouteDTO.ProcessStepDTO.fromEntity(step)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/{id}/first-step")
    public ResponseEntity<ProcessRouteDTO.ProcessStepDTO> getFirstStep(@PathVariable Long id) {
        return processRouteRepository.findById(id)
            .flatMap(ProcessRoute::getFirstStep)
            .map(step -> ResponseEntity.ok(ProcessRouteDTO.ProcessStepDTO.fromEntity(step)))
            .orElse(ResponseEntity.notFound().build());
    }
}