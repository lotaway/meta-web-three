package com.metawebthree.mes.adapter.http.controller;

import com.metawebthree.mes.domain.entity.ParameterGroupTemplate;
import com.metawebthree.mes.domain.repository.ParameterGroupTemplateRepository;
import com.metawebthree.mes.interfaces.dto.ParameterGroupTemplateDTO;
import com.metawebthree.mes.domain.entity.ProcessParameter;
import com.metawebthree.mes.domain.repository.ProcessParameterRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/parameter-group-template")
@RequiredArgsConstructor
public class ParameterGroupTemplateController {
    
    private final ParameterGroupTemplateRepository templateRepository;
    private final ProcessParameterRepository processParameterRepository;
    
    /**
     * 创建参数组模板
     */
    @PostMapping
    public ResponseEntity<ParameterGroupTemplateDTO> create(
            @RequestBody ParameterGroupTemplateDTO.CreateRequest request) {
        
        // 检查模板编码是否已存在
        if (templateRepository.existsByTemplateCode(request.getTemplateCode())) {
            return ResponseEntity.badRequest().build();
        }
        
        ParameterGroupTemplate template = request.toEntity();
        
        // 如果有参数ID，批量获取对应的参数编码
        if (request.getParameterIds() != null && !request.getParameterIds().isEmpty()) {
            List<ProcessParameter> params = processParameterRepository.findByIds(request.getParameterIds());
            List<String> codes = params.stream()
                    .map(ProcessParameter::getParamCode)
                    .collect(Collectors.toList());
            template.setParameterCodes(codes);
        }
        
        ParameterGroupTemplate saved = templateRepository.save(template);
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(saved));
    }
    
    /**
     * 更新参数组模板
     */
    @PutMapping("/{id}")
    public ResponseEntity<ParameterGroupTemplateDTO> update(
            @PathVariable Long id,
            @RequestBody ParameterGroupTemplateDTO.UpdateRequest request) {
        
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        if (request.getTemplateName() != null) {
            template.setTemplateName(request.getTemplateName());
        }
        if (request.getProductType() != null) {
            template.setProductType(request.getProductType());
        }
        if (request.getDescription() != null) {
            template.setDescription(request.getDescription());
        }
        if (request.getDisplayOrder() != null) {
            template.setDisplayOrder(request.getDisplayOrder());
        }
        if (request.getParameterIds() != null) {
            template.setParameterIds(request.getParameterIds());
            // 批量获取对应的参数编码
            if (!request.getParameterIds().isEmpty()) {
                List<ProcessParameter> params = processParameterRepository.findByIds(request.getParameterIds());
                List<String> codes = params.stream()
                        .map(ProcessParameter::getParamCode)
                        .collect(Collectors.toList());
                template.setParameterCodes(codes);
            } else {
                template.setParameterCodes(List.of());
            }
        }
        
        templateRepository.save(template);
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
    
    /**
     * 删除参数组模板
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        templateRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
    
    /**
     * 根据ID获取参数组模板
     */
    @GetMapping("/{id}")
    public ResponseEntity<ParameterGroupTemplateDTO> getById(@PathVariable Long id) {
        return templateRepository.findById(id)
            .map(template -> ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    /**
     * 根据模板编码获取参数组模板
     */
    @GetMapping("/code/{templateCode}")
    public ResponseEntity<ParameterGroupTemplateDTO> getByCode(@PathVariable String templateCode) {
        return templateRepository.findByTemplateCode(templateCode)
            .map(template -> ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template)))
            .orElse(ResponseEntity.notFound().build());
    }
    
    /**
     * 根据产品类型获取参数组模板列表
     */
    @GetMapping("/product-type/{productType}")
    public ResponseEntity<List<ParameterGroupTemplateDTO>> getByProductType(
            @PathVariable String productType) {
        
        List<ParameterGroupTemplateDTO> templates = templateRepository.findByProductType(productType)
            .stream()
            .map(ParameterGroupTemplateDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(templates);
    }
    
    /**
     * 根据产品类型获取激活的参数组模板
     */
    @GetMapping("/product-type/{productType}/active")
    public ResponseEntity<List<ParameterGroupTemplateDTO>> getActiveByProductType(
            @PathVariable String productType) {
        
        List<ParameterGroupTemplateDTO> templates = templateRepository
            .findByProductTypeAndStatus(productType, ParameterGroupTemplate.TemplateStatus.ACTIVE)
            .stream()
            .map(ParameterGroupTemplateDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(templates);
    }
    
    /**
     * 获取所有参数组模板
     */
    @GetMapping
    public ResponseEntity<List<ParameterGroupTemplateDTO>> list(
            @RequestParam(required = false) String status) {
        
        List<ParameterGroupTemplate> templates;
        if (status != null && !status.isEmpty()) {
            ParameterGroupTemplate.TemplateStatus templateStatus = 
                ParameterGroupTemplate.TemplateStatus.valueOf(status);
            templates = templateRepository.findByStatus(templateStatus);
        } else {
            templates = templateRepository.findAll();
        }
        
        List<ParameterGroupTemplateDTO> dtos = templates.stream()
            .map(ParameterGroupTemplateDTO::fromEntity)
            .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    /**
     * 激活参数组模板
     */
    @PostMapping("/{id}/activate")
    public ResponseEntity<ParameterGroupTemplateDTO> activate(@PathVariable Long id) {
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        template.activate();
        templateRepository.save(template);
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
    
    /**
     * 停用参数组模板
     */
    @PostMapping("/{id}/deactivate")
    public ResponseEntity<ParameterGroupTemplateDTO> deactivate(@PathVariable Long id) {
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        template.deactivate();
        templateRepository.save(template);
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
    
    /**
     * 归档参数组模板
     */
    @PostMapping("/{id}/archive")
    public ResponseEntity<ParameterGroupTemplateDTO> archive(@PathVariable Long id) {
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        template.archive();
        templateRepository.save(template);
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
    
    /**
     * 添加参数到模板
     */
    @PostMapping("/{id}/parameters")
    public ResponseEntity<ParameterGroupTemplateDTO> addParameter(
            @PathVariable Long id,
            @RequestParam Long parameterId) {
        
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        ProcessParameter param = processParameterRepository.findById(parameterId)
            .orElseThrow(() -> new RuntimeException("Parameter not found: " + parameterId));
        
        template.addParameter(parameterId, param.getParamCode());
        templateRepository.save(template);
        
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
    
    /**
     * 从模板移除参数
     */
    @DeleteMapping("/{id}/parameters/{parameterId}")
    public ResponseEntity<ParameterGroupTemplateDTO> removeParameter(
            @PathVariable Long id,
            @PathVariable Long parameterId) {
        
        ParameterGroupTemplate template = templateRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Template not found: " + id));
        
        template.removeParameter(parameterId);
        templateRepository.save(template);
        
        return ResponseEntity.ok(ParameterGroupTemplateDTO.fromEntity(template));
    }
}