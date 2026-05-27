package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class SopDocumentTest {
    
    @Test
    void testCreateSopDocument() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        assertEquals("SOP-001", doc.getDocumentCode());
        assertEquals("组装SOP", doc.getDocumentName());
        assertEquals("PDF", doc.getDocumentType());
        assertEquals("组装工艺", doc.getCategory());
        assertEquals(0, doc.getCurrentVersionNo());
        assertEquals(SopDocument.SopStatus.DRAFT, doc.getStatus());
        assertNotNull(doc.getCreatedAt());
        assertNotNull(doc.getUpdatedAt());
    }
    
    @Test
    void testActivateSopDocument() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        assertEquals(SopDocument.SopStatus.DRAFT, doc.getStatus());
        doc.activate();
        assertEquals(SopDocument.SopStatus.ACTIVE, doc.getStatus());
    }
    
    @Test
    void testArchiveSopDocument() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        doc.activate();
        
        assertEquals(SopDocument.SopStatus.ACTIVE, doc.getStatus());
        doc.archive();
        assertEquals(SopDocument.SopStatus.ARCHIVED, doc.getStatus());
    }
    
    @Test
    void testAddVersion() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        SopDocumentVersion version = doc.addVersion("sop-v1.pdf", "/files/sop-v1.pdf", "admin", "初始版本");
        
        assertNotNull(version);
        assertEquals(1, version.getVersionNo());
        assertEquals("sop-v1.pdf", version.getFileName());
        assertEquals("PDF", version.getFileType());
        assertEquals("admin", version.getUploader());
        assertEquals("初始版本", version.getChangeDescription());
        assertTrue(version.getIsCurrentVersion());
        assertEquals(1, doc.getCurrentVersionNo());
    }
    
    @Test
    void testAddMultipleVersions() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        doc.addVersion("sop-v1.pdf", "/files/sop-v1.pdf", "admin", "v1");
        doc.addVersion("sop-v2.pdf", "/files/sop-v2.pdf", "admin", "v2");
        
        assertEquals(2, doc.getCurrentVersionNo());
        
        SopDocumentVersion current = doc.getCurrentVersion();
        assertNotNull(current);
        assertEquals(2, current.getVersionNo());
        assertEquals("sop-v2.pdf", current.getFileName());
    }
    
    @Test
    void testGetCurrentVersion() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        assertNull(doc.getCurrentVersion());
        
        doc.addVersion("sop-v1.pdf", "/files/sop-v1.pdf", "admin", "v1");
        
        SopDocumentVersion current = doc.getCurrentVersion();
        assertNotNull(current);
        assertEquals("sop-v1.pdf", current.getFileName());
    }
    
    @Test
    void testGetVersionHistory() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        doc.addVersion("sop-v1.pdf", "/files/sop-v1.pdf", "admin", "v1");
        doc.addVersion("sop-v2.pdf", "/files/sop-v2.pdf", "admin", "v2");
        
        List<SopDocumentVersion> history = doc.getVersionHistory();
        assertEquals(2, history.size());
        assertEquals(2, history.get(0).getVersionNo());
        assertEquals(1, history.get(1).getVersionNo());
    }
    
    @Test
    void testBindRoute() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        doc.bindRoute("ROUTE-001", "产品A工艺", 1, "PC-001", "组装", "WS-001", "组装工位");
        
        assertNotNull(doc.getRouteBindings());
        assertEquals(1, doc.getRouteBindings().size());
        
        SopRouteBinding binding = doc.getRouteBindings().get(0);
        assertEquals("ROUTE-001", binding.getRouteCode());
        assertEquals(1, binding.getStepNo());
        assertEquals("PC-001", binding.getProcessCode());
        assertEquals("WS-001", binding.getWorkstationId());
        assertTrue(binding.getIsActive());
    }
    
    @Test
    void testUnbindRoute() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        doc.bindRoute("ROUTE-001", "产品A工艺", 1, "PC-001", "组装", "WS-001", "组装工位");
        doc.bindRoute("ROUTE-001", "产品A工艺", 2, "PC-002", "测试", "WS-002", "测试工位");
        
        assertEquals(2, doc.getRouteBindings().size());
        
        doc.unbindRoute("ROUTE-001", 1);
        
        assertEquals(1, doc.getRouteBindings().size());
        assertEquals(2, doc.getRouteBindings().get(0).getStepNo());
    }
    
    @Test
    void testUnbindRouteAllSteps() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        doc.bindRoute("ROUTE-001", "产品A工艺", 1, "PC-001", "组装", "WS-001", "组装工位");
        doc.bindRoute("ROUTE-001", "产品A工艺", 2, "PC-002", "测试", "WS-002", "测试工位");
        
        doc.unbindRoute("ROUTE-001", null);
        
        assertEquals(0, doc.getRouteBindings().size());
    }
    
    @Test
    void testFullLifecycle() {
        SopDocument doc = new SopDocument();
        
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        assertEquals(SopDocument.SopStatus.DRAFT, doc.getStatus());
        
        doc.activate();
        assertEquals(SopDocument.SopStatus.ACTIVE, doc.getStatus());
        
        doc.addVersion("sop-v1.pdf", "/files/sop-v1.pdf", "admin", "初始版本");
        assertEquals(1, doc.getCurrentVersionNo());
        
        doc.bindRoute("ROUTE-001", "产品A工艺", 1, "PC-001", "组装", "WS-001", "组装工位");
        assertEquals(1, doc.getRouteBindings().size());
        
        doc.archive();
        assertEquals(SopDocument.SopStatus.ARCHIVED, doc.getStatus());
    }
    
    @Test
    void testVersionFileTypeExtraction() {
        SopDocument doc = new SopDocument();
        doc.create("SOP-001", "组装SOP", "PDF", "组装工艺");
        
        SopDocumentVersion version1 = doc.addVersion("manual.pdf", "/files/manual.pdf", "admin", "v1");
        assertEquals("PDF", version1.getFileType());
        
        SopDocumentVersion version2 = doc.addVersion("guide.docx", "/files/guide.docx", "admin", "v2");
        assertEquals("DOCX", version2.getFileType());
        
        SopDocumentVersion version3 = doc.addVersion("instruction", "/files/instruction", "admin", "v3");
        assertEquals("UNKNOWN", version3.getFileType());
    }
}