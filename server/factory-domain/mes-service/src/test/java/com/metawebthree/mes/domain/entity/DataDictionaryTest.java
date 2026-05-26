package com.metawebthree.mes.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

class DataDictionaryTest {
    
    @Test
    void createShouldInitializeDictionaryWithDefaultValues() {
        DataDictionary dict = DataDictionary.create("GENDER", "性别", "性别字典");
        
        assertEquals("GENDER", dict.getDictCode());
        assertEquals("性别", dict.getDictName());
        assertEquals("性别字典", dict.getDescription());
        assertEquals(DataDictionary.DictStatus.ACTIVE, dict.getStatus());
        assertNotNull(dict.getItems());
        assertTrue(dict.getItems().isEmpty());
        assertNotNull(dict.getCreatedAt());
    }
    
    @Test
    void addItemShouldAddItemToDictionary() {
        DataDictionary dict = DataDictionary.create("GENDER", "性别", "性别字典");
        
        DataDictionary.DataDictionaryItem item = dict.addItem("M", "男", 1);
        
        assertEquals("M", item.getItemCode());
        assertEquals("男", item.getItemLabel());
        assertEquals(1, item.getSortOrder());
        assertEquals(DataDictionary.DataDictionaryItem.ItemStatus.ACTIVE, item.getStatus());
        assertEquals(1, dict.getItems().size());
    }
    
    @Test
    void removeItemShouldRemoveItemFromDictionary() {
        DataDictionary dict = DataDictionary.create("GENDER", "性别", "性别字典");
        dict.addItem("M", "男", 1);
        dict.addItem("F", "女", 2);
        
        dict.removeItem("M");
        
        assertEquals(1, dict.getItems().size());
        assertTrue(dict.getItems().stream().noneMatch(i -> i.getItemCode().equals("M")));
    }
    
    @Test
    void getActiveItemsShouldReturnOnlyActiveItems() {
        DataDictionary dict = DataDictionary.create("GENDER", "性别", "性别字典");
        DataDictionary.DataDictionaryItem male = dict.addItem("M", "男", 1);
        DataDictionary.DataDictionaryItem female = dict.addItem("F", "女", 2);
        
        female.setStatus(DataDictionary.DataDictionaryItem.ItemStatus.INACTIVE);
        
        List<DataDictionary.DataDictionaryItem> activeItems = dict.getActiveItems();
        
        assertEquals(1, activeItems.size());
        assertEquals("M", activeItems.get(0).getItemCode());
    }
}