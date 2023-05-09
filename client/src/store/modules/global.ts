import {createSlice} from '@reduxjs/toolkit';
import type {PayloadAction} from '@reduxjs/toolkit';

export interface GlobalStates {
    clickCount: number
    pageErrorTitle: string
    pageMsgListTitle: string
}

const initialState: GlobalStates = {
    clickCount: 0,
    pageErrorTitle: "找不到页面",
    pageMsgListTitle: "我的消息"
};
const globalSlice = createSlice({
    name: "global",
    initialState,
    reducers: {
        //  @todo sync to other app/webview/micro-service store.
        sendSync: (state, action: PayloadAction<null>) => {

        },
        //  @todo receive store from other app/webview/micron-service, update current store.
        receiveSync: (state, action: PayloadAction<typeof state>) => {
        },
        setPageTitle: (state, action: PayloadAction<string>) => {
            typeof document !== "undefined" && (document.title = action.payload);
        },
        changeClickCount: (state, action: PayloadAction<number>) => {
            state.clickCount++;
        }
    }
});

export const {sendSync, receiveSync, setPageTitle, changeClickCount} = globalSlice.actions;
export default globalSlice.reducer;
