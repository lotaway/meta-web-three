package com.metawebthree.mes.domain.service;

import com.metawebthree.mes.domain.entity.SopDocument;
import com.metawebthree.mes.domain.entity.SopDocumentVersion;
import com.metawebthree.mes.domain.repository.SopDocumentRepository;
import com.metawebthree.mes.domain.repository.SopRouteBindingRepository;

import java.util.List;
import java.util.Optional;

public class SopDocumentServiceImpl implements SopDocumentService {

    private final SopDocumentRepository repository;
    private final SopRouteBindingRepository routeBindingRepository;

    public SopDocumentServiceImpl(SopDocumentRepository repository, 
                                   SopRouteBindingRepository routeBindingRepository) {
        this.repository = repository;
        this.routeBindingRepository = routeBindingRepository;
    }

    @Override
    public SopDocument create(SopDocument sopDocument) {
        if (repository.existsByDocumentCode(sopDocument.getDocumentCode())) {
            throw new IllegalArgumentException("SOP文档编码已存在: " + sopDocument.getDocumentCode());
        }
        return repository.save(sopDocument);
    }

    @Override
    public Optional<SopDocument> findById(Long id) {
        return repository.findById(id);
    }

    @Override
    public Optional<SopDocument> findByDocumentCode(String documentCode) {
        return repository.findByDocumentCode(documentCode);
    }

    @Override
    public List<SopDocument> findAll() {
        return repository.findAll();
    }

    @Override
    public List<SopDocument> findByStatus(SopDocument.SopStatus status) {
        return repository.findByStatus(status);
    }

    @Override
    public List<SopDocument> findByDocumentType(String documentType) {
        return repository.findByDocumentType(documentType);
    }

    @Override
    public SopDocument update(SopDocument sopDocument) {
        return repository.save(sopDocument);
    }

    @Override
    public void deleteById(Long id) {
        repository.deleteById(id);
    }

    @Override
    public SopDocument activate(Long id) {
        SopDocument doc = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("SOP文档不存在: " + id));
        doc.activate();
        return repository.save(doc);
    }

    @Override
    public SopDocument archive(Long id) {
        SopDocument doc = repository.findById(id)
            .orElseThrow(() -> new IllegalArgumentException("SOP文档不存在: " + id));
        doc.archive();
        return repository.save(doc);
    }

    @Override
    public SopDocumentVersion addVersion(Long sopDocumentId, String fileName, 
                                                      String filePath, String uploader, 
                                                      String changeDescription) {
        SopDocument doc = repository.findById(sopDocumentId)
            .orElseThrow(() -> new IllegalArgumentException("SOP文档不存在: " + sopDocumentId));
        
        SopDocumentVersion newVersion = doc.addVersion(fileName, filePath, uploader, changeDescription);
        repository.save(doc);
        return newVersion;
    }

    @Override
    public SopDocument bindRoute(Long sopDocumentId, String routeCode, String routeName, 
                                  Integer stepNo, String processCode, String processName,
                                  Long workstationId, String workstationName) {
        SopDocument doc = repository.findById(sopDocumentId)
            .orElseThrow(() -> new IllegalArgumentException("SOP文档不存在: " + sopDocumentId));
        
        doc.bindRoute(routeCode, routeName, stepNo, processCode, processName, workstationId, workstationName);
        return repository.save(doc);
    }

    @Override
    public SopDocument unbindRoute(Long sopDocumentId, String routeCode, Integer stepNo) {
        SopDocument doc = repository.findById(sopDocumentId)
            .orElseThrow(() -> new IllegalArgumentException("SOP文档不存在: " + sopDocumentId));
        
        doc.unbindRoute(routeCode, stepNo);
        return repository.save(doc);
    }

    @Override
    public List<SopDocument> findByRouteAndStep(String routeCode, Integer stepNo) {
        return routeBindingRepository.findByRouteCodeAndStepNo(routeCode, stepNo);
    }

    @Override
    public List<SopDocument> findByWorkstation(String workstationId) {
        return routeBindingRepository.findByWorkstationId(workstationId);
    }
}