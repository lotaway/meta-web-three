package com.metawebthree.live.application;

import com.metawebthree.live.domain.model.*;
import com.metawebthree.live.domain.repository.*;
import com.metawebthree.live.domain.ports.OrderPort;
import com.metawebthree.live.domain.ports.ProductPort;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class LiveApplicationServiceTest {

    @Mock
    private AnchorRepository anchorRepository;
    @Mock
    private LiveRoomRepository liveRoomRepository;
    @Mock
    private LiveProductRepository liveProductRepository;
    @Mock
    private LiveCommentRepository liveCommentRepository;
    @Mock
    private LiveOrderRepository liveOrderRepository;
    @Mock
    private OrderPort orderPort;
    @Mock
    private ProductPort productPort;

    private LiveApplicationService service;

    @BeforeEach
    void setUp() {
        service = new LiveApplicationService(
            anchorRepository,
            liveRoomRepository,
            liveProductRepository,
            liveCommentRepository,
            liveOrderRepository,
            orderPort,
            productPort
        );
    }

    @Test
    void createAnchor_shouldSaveAndReturnAnchor() {
        Anchor inputAnchor = new Anchor();
        inputAnchor.setUserId(100L);
        inputAnchor.setAnchorName("TestAnchor");
        inputAnchor.setAvatar("avatar.jpg");
        inputAnchor.setDescription("Test description");

        Anchor savedAnchor = new Anchor();
        savedAnchor.setId(1L);
        savedAnchor.setUserId(100L);
        savedAnchor.setAnchorName("TestAnchor");
        savedAnchor.setAvatar("avatar.jpg");
        savedAnchor.setDescription("Test description");
        savedAnchor.setFollowerCount(0);
        savedAnchor.setStatus(1);

        when(anchorRepository.save(any(Anchor.class))).thenReturn(savedAnchor);

        Anchor result = service.createAnchor(100L, "TestAnchor", "avatar.jpg", "Test description");

        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals("TestAnchor", result.getAnchorName());
        verify(anchorRepository).save(any(Anchor.class));
    }

    @Test
    void startLiveRoom_shouldCreateRoomWhenAnchorExists() {
        Anchor anchor = new Anchor();
        anchor.setId(1L);
        anchor.setUserId(100L);

        LiveRoom savedRoom = new LiveRoom();
        savedRoom.setId(10L);
        savedRoom.setAnchorId(1L);
        savedRoom.setRoomName("TestRoom");
        savedRoom.setViewerCount(0);
        savedRoom.setStatus(1);

        when(anchorRepository.findById(1L)).thenReturn(anchor);
        when(liveRoomRepository.save(any(LiveRoom.class))).thenReturn(savedRoom);

        LiveRoom result = service.startLiveRoom(1L, "TestRoom", "cover.jpg", "Room description");

        assertNotNull(result);
        assertEquals(10L, result.getId());
        assertEquals("TestRoom", result.getRoomName());
        verify(liveRoomRepository).save(any(LiveRoom.class));
    }

    @Test
    void startLiveRoom_shouldThrowWhenAnchorNotFound() {
        when(anchorRepository.findById(1L)).thenReturn(null);

        assertThrows(IllegalArgumentException.class, () -> 
            service.startLiveRoom(1L, "TestRoom", "cover.jpg", "desc")
        );
    }

    @Test
    void createLiveOrder_shouldCreateOrderSuccessfully() {
        LiveProduct product = new LiveProduct();
        product.setId(100L);
        product.setProductId(200L);
        product.setDiscountPrice(5000L);
        product.setRoomId(10L);

        LiveOrder savedOrder = new LiveOrder();
        savedOrder.setId(1L);
        savedOrder.setRoomId(10L);
        savedOrder.setProductId(200L);
        savedOrder.setUserId(300L);
        savedOrder.setOrderId(500L);
        savedOrder.setQuantity(2);
        savedOrder.setTotalAmount(10000L);
        savedOrder.setStatus(1);

        when(liveProductRepository.findById(100L)).thenReturn(product);
        when(productPort.reduceStock(200L, 2)).thenReturn(true);
        when(orderPort.createOrder(300L, 200L, 2, 10L)).thenReturn(500L);
        when(liveOrderRepository.save(any(LiveOrder.class))).thenReturn(savedOrder);

        LiveOrder result = service.createLiveOrder(10L, 100L, 300L, 2);

        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(500L, result.getOrderId());
        assertEquals(10000L, result.getTotalAmount());
        verify(productPort).reduceStock(200L, 2);
        verify(orderPort).createOrder(300L, 200L, 2, 10L);
    }

    @Test
    void createLiveOrder_shouldThrowWhenProductNotFound() {
        when(liveProductRepository.findById(100L)).thenReturn(null);

        assertThrows(IllegalArgumentException.class, () ->
            service.createLiveOrder(10L, 100L, 300L, 2)
        );
    }

    @Test
    void createLiveOrder_shouldThrowWhenInsufficientStock() {
        LiveProduct product = new LiveProduct();
        product.setId(100L);
        product.setProductId(200L);

        when(liveProductRepository.findById(100L)).thenReturn(product);
        when(productPort.reduceStock(200L, 2)).thenReturn(false);

        assertThrows(IllegalStateException.class, () ->
            service.createLiveOrder(10L, 100L, 300L, 2)
        );
    }

    @Test
    void attachProduct_shouldAttachProductToRoom() {
        LiveRoom room = new LiveRoom();
        room.setId(10L);

        LiveProduct savedProduct = new LiveProduct();
        savedProduct.setId(1L);
        savedProduct.setRoomId(10L);
        savedProduct.setProductId(200L);
        savedProduct.setPrice(6000L);
        savedProduct.setDiscountPrice(5000L);
        savedProduct.setStock(100);
        savedProduct.setStatus(1);

        when(liveRoomRepository.findById(10L)).thenReturn(room);
        when(liveProductRepository.save(any(LiveProduct.class))).thenReturn(savedProduct);

        LiveProduct result = service.attachProduct(10L, 200L, 6000L, 5000L, 100);

        assertNotNull(result);
        assertEquals(1L, result.getId());
        assertEquals(200L, result.getProductId());
    }

    @Test
    void postComment_shouldPostCommentSuccessfully() {
        LiveRoom room = new LiveRoom();
        room.setId(10L);

        LiveComment savedComment = new LiveComment();
        savedComment.setId(1L);
        savedComment.setRoomId(10L);
        savedComment.setUserId(100L);
        savedComment.setUserName("User1");
        savedComment.setContent("Great!");
        savedComment.setType(1);

        when(liveRoomRepository.findById(10L)).thenReturn(room);
        when(liveCommentRepository.save(any(LiveComment.class))).thenReturn(savedComment);

        LiveComment result = service.postComment(10L, 100L, "User1", "Great!");

        assertNotNull(result);
        assertEquals("Great!", result.getContent());
        assertEquals(1, result.getType());
    }

    @Test
    void getLiveRoomsByAnchorId_shouldReturnRoomList() {
        LiveRoom room1 = new LiveRoom();
        room1.setId(10L);
        LiveRoom room2 = new LiveRoom();
        room2.setId(11L);

        when(liveRoomRepository.findByAnchorId(1L)).thenReturn(List.of(room1, room2));

        List<LiveRoom> result = service.getLiveRoomsByAnchorId(1L);

        assertEquals(2, result.size());
    }
}