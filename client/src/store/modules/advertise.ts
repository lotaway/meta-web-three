import {createSlice, createAsyncThunk} from "@reduxjs/toolkit";
import {NSAdvertisement} from "../../services/advertisement";
import SystemImpl from "../../system/SystemImpl";

export interface States {

}

const initialState: States = {

    },
    serviceAdvertise = new NSAdvertisement.Service(SystemImpl.getInstance()),
    name = "advertise",
    advertiseSlice = createSlice({
        name,
        initialState,
        reducers: {
            setHomeBanner(state: States, {payload}) {

            }
        },
        extraReducers: (builder) => {
            builder
                .addCase(getHomeBanner.fulfilled.type, (state, action: any) => {

                })
        }
    });
export const getHomeBanner = createAsyncThunk<unknown, {
    pageIndex?: number
} | undefined>(
    `${name}/getHomeBanner`,
    async () => {
        return await serviceAdvertise.getAppHomeBanner()
    })
export default advertiseSlice.reducer;
