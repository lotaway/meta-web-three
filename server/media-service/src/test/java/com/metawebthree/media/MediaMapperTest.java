package com.metawebthree.media;

import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.junit.jupiter.api.Disabled;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.core.toolkit.LambdaUtils;
import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkDO;
import com.metawebthree.media.DO.PeopleDO;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree.media",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
public class MediaMapperTest {

    @Autowired
    private ArtWorkMapper artWorkMapper;

    @Autowired
    private PeopleMapper peopleMapper;

    @Test
    @Disabled
    public void fixArtWorkDirector() {
        String directorName = "文舟";
        List<PeopleDO> peopleDOS = peopleMapper.selectByMap(Map.of("name", directorName));
        if (peopleDOS.size() == 0) {
            UpdateWrapper<PeopleDO> peopleUpdateWrapper = new UpdateWrapper<>();
            peopleUpdateWrapper.eq("name", directorName).set("types", new Short[] { Short.valueOf("1") });
            int result = peopleMapper.update(peopleUpdateWrapper);
            Assert.assertEquals(1, result);
            MPJLambdaWrapper<PeopleDO> wrapper = new MPJLambdaWrapper<>();
            wrapper.select(PeopleDO::getId).eq(PeopleDO::getName, directorName);
            PeopleDO peopleDO = peopleMapper.selectOne(wrapper);
            UpdateWrapper<ArtWorkDO> updateWrapper2 = new UpdateWrapper<>();
            updateWrapper2.eq("series", "骑士的沙丘").set("director", peopleDO.getId());
            int result2 = artWorkMapper.update(updateWrapper2);
            Assert.assertTrue(result2 > 0);
        }
        Assert.assertNotNull(artWorkMapper.selectById(11));
    }

    @Test
    public void fixSubtitleError() {
        ArtWorkDO artWorkDO = ArtWorkDO.builder().subtitle("undefined ").build();
        UpdateWrapper<ArtWorkDO> updateWrapper = new UpdateWrapper<>();
        String filedName = "subtitle";
        updateWrapper.likeRight(filedName, artWorkDO.getSubtitle())
                .setSql(String.format("%s = REPLACE(%s, 'undefined ', '')", filedName, filedName));
        int result = artWorkMapper.update(updateWrapper);
        Assert.assertNotNull(result);
    }
}
