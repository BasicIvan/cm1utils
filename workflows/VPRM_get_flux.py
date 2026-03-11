import xarray as xr
import pandas as pd
import numpy as np


def aave(*sectors):
    sectors_aaveraged = np.zeros(len(sectors))
    for i, s in enumerate(sectors):
        sectors_aaveraged[i] = np.average(s, (0, 1))
    return sectors_aaveraged


def emission_hour(df_co2, df_ch4, aa_co2, aa_ch4, ib, ie):
    """
    Compute hourly total emissions (area-averaged) for CO2 and CH4.

    aa_co2, aa_ch4: dict with keys:
      traffic, residential, services, industry, agriculture
    """
    co2_hour = np.zeros(ie - ib)
    ch4_hour = np.zeros(ie - ib)

    for j, i in enumerate(range(ib, ie)):
        co2_hour[j] = (
            df_co2.iloc[i]["CO2_traffic"] * aa_co2["traffic"]
            + df_co2.iloc[i]["CO2_residential"] * aa_co2["residential"]
            + df_co2.iloc[i]["CO2_services"] * aa_co2["services"]
            + df_co2.iloc[i]["CO2_industry"] * aa_co2["industry"]
            + df_co2.iloc[i]["CO2_agriculture"] * aa_co2["agriculture"]
        )

        ch4_hour[j] = (
            df_ch4.iloc[i]["CH4_traffic"] * aa_ch4["traffic"]
            + df_ch4.iloc[i]["CH4_residential"] * aa_ch4["residential"]
            + df_ch4.iloc[i]["CH4_services"] * aa_ch4["services"]
            + df_ch4.iloc[i]["CH4_industry"] * aa_ch4["industry"]
            + df_ch4.iloc[i]["CH4_agriculture"] * aa_ch4["agriculture"]
        )

    return co2_hour, ch4_hour


def build_anthro_hourly(
    dir_path,
    co2_nc="MeteotestEKAT_2015_CO2.nc",
    ch4_nc="MeteotestEKAT_2015_CH4.nc",
    co2_csv="MeteotestEKAT_2015_CO2_timeprofile.csv",
    ch4_csv="MeteotestEKAT_2015_CH4_timeprofile.csv",
    begin="2021-07-21 00:00:00",
    end="2021-07-22 00:00:00",
):
    # load area emissions
    dco2 = xr.open_dataset(f"{dir_path}/{co2_nc}")
    dch4 = xr.open_dataset(f"{dir_path}/{ch4_nc}")

    aa_co2 = dict(
        zip(
            ["traffic", "residential", "services", "industry", "agriculture"],
            aave(
                dco2.CO2_traffic.values,
                dco2.CO2_residential.values,
                dco2.CO2_services.values,
                dco2.CO2_industry.values,
                dco2.CO2_agriculture.values,
            ),
        )
    )

    aa_ch4 = dict(
        zip(
            ["traffic", "residential", "services", "industry", "agriculture"],
            aave(
                dch4.CH4_traffic.values,
                dch4.CH4_residential.values,
                dch4.CH4_services.values,
                dch4.CH4_industry.values,
                dch4.CH4_agriculture.values,
            ),
        )
    )

    # load time profiles
    df = pd.read_csv(f"{dir_path}/{co2_csv}", delimiter=",")
    df1 = pd.read_csv(f"{dir_path}/{ch4_csv}", delimiter=",")

    ib = df.index[df["dtm"] == begin].item()
    ie = df.index[df["dtm"] == end].item()

    co2_hour, ch4_hour = emission_hour(df, df1, aa_co2, aa_ch4, ib, ie)
    return co2_hour, ch4_hour


def build_biogenic(
    dir_path,
    vprm_nc="VPRM_CoCO2_BRM_20210101-20211231.nc",
    begin_np64=np.datetime64("2021-07-21T00:00:00"),
    end_np64=np.datetime64("2021-07-22T00:00:00"),
):
    vprm = xr.open_dataset(f"{dir_path}/{vprm_nc}")

    dtm = vprm.time.values
    ib = np.where(dtm == begin_np64)[0][0]
    ie = np.where(dtm == end_np64)[0][0]

    resp = vprm.RESP.values[ib:ie]
    gpp = vprm.GPP.values[ib:ie]
    return resp, gpp


def main():
    dir_path = "/work/bb1096/b381871/fluxes"
    co2_hour, ch4_hour = build_anthro_hourly(dir_path)
    resp, gpp = build_biogenic(dir_path)

    # do something with results (print/save/plot)
    print(co2_hour.shape, ch4_hour.shape, resp.shape, gpp.shape)


if __name__ == "__main__":
    main()