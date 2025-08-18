package com.metawebthree.service;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.*;

class MockitoTest {

    @Test
    void testMockitoWorks() {
        String mockString = Mockito.mock(String.class);
        Mockito.when(mockString.length()).thenReturn(5);
        
        assertEquals(5, mockString.length());
        assertTrue(true);
    }
}
