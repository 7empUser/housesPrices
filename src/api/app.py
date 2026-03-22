from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from src.model import predict

app = FastAPI()

class HouseInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Numerical Features
    lot_frontage:    Optional[float] = Field(default=None, alias='LotFrontage')
    lot_area:        Optional[float] = Field(default=None, alias='LotArea')
    overall_qual:    Optional[float] = Field(default=None, alias='OverallQual')
    overall_cond:    Optional[float] = Field(default=None, alias='OverallCond')
    year_built:      Optional[float] = Field(default=None, alias='YearBuilt')
    year_remod_add:  Optional[float] = Field(default=None, alias='YearRemodAdd')
    mas_vnr_area:    Optional[float] = Field(default=None, alias='MasVnrArea')
    bsmt_fin_sf1:    Optional[float] = Field(default=None, alias='BsmtFinSF1')
    bsmt_fin_sf2:    Optional[float] = Field(default=None, alias='BsmtFinSF2')
    bsmt_unf_sf:     Optional[float] = Field(default=None, alias='BsmtUnfSF')
    total_bsmt_sf:   Optional[float] = Field(default=None, alias='TotalBsmtSF')
    first_flr_sf:    Optional[float] = Field(default=None, alias='1stFlrSF')
    second_flr_sf:   Optional[float] = Field(default=None, alias='2ndFlrSF')
    low_qual_fin_sf: Optional[float] = Field(default=None, alias='LowQualFinSF')
    gr_liv_area:     Optional[float] = Field(default=None, alias='GrLivArea')
    bsmt_full_bath:  Optional[float] = Field(default=None, alias='BsmtFullBath')
    bsmt_half_bath:  Optional[float] = Field(default=None, alias='BsmtHalfBath')
    full_bath:       Optional[float] = Field(default=None, alias='FullBath')
    half_bath:       Optional[float] = Field(default=None, alias='HalfBath')
    bedroom_abv_gr:  Optional[float] = Field(default=None, alias='BedroomAbvGr')
    kitchen_abv_gr:  Optional[float] = Field(default=None, alias='KitchenAbvGr')
    tot_rms_abv_grd: Optional[float] = Field(default=None, alias='TotRmsAbvGrd')
    fireplaces:      Optional[float] = Field(default=None, alias='Fireplaces')
    garage_yr_blt:   Optional[float] = Field(default=None, alias='GarageYrBlt')
    garage_cars:     Optional[float] = Field(default=None, alias='GarageCars')
    garage_area:     Optional[float] = Field(default=None, alias='GarageArea')
    wood_deck_sf:    Optional[float] = Field(default=None, alias='WoodDeckSF')
    open_porch_sf:   Optional[float] = Field(default=None, alias='OpenPorchSF')
    enclosed_porch:  Optional[float] = Field(default=None, alias='EnclosedPorch')
    ssn_porch:       Optional[float] = Field(default=None, alias='3SsnPorch')
    screen_porch:    Optional[float] = Field(default=None, alias='ScreenPorch')
    pool_area:       Optional[float] = Field(default=None, alias='PoolArea')
    misc_val:        Optional[float] = Field(default=None, alias='MiscVal')
    mo_sold:         Optional[float] = Field(default=None, alias='MoSold')
    yr_sold:         Optional[float] = Field(default=None, alias='YrSold')
    ms_sub_class:    Optional[float] = Field(default=None, alias='MSSubClass')

    # Categorical Features 
    ms_zoning:       Optional[str] = Field(default=None, alias='MSZoning')
    street:          Optional[str] = Field(default=None, alias='Street')
    alley:           Optional[str] = Field(default=None, alias='Alley')
    lot_shape:       Optional[str] = Field(default=None, alias='LotShape')
    land_contour:    Optional[str] = Field(default=None, alias='LandContour')
    utilities:       Optional[str] = Field(default=None, alias='Utilities')
    lot_config:      Optional[str] = Field(default=None, alias='LotConfig')
    land_slope:      Optional[str] = Field(default=None, alias='LandSlope')
    neighborhood:    Optional[str] = Field(default=None, alias='Neighborhood')
    condition1:      Optional[str] = Field(default=None, alias='Condition1')
    condition2:      Optional[str] = Field(default=None, alias='Condition2')
    bldg_type:       Optional[str] = Field(default=None, alias='BldgType')
    house_style:     Optional[str] = Field(default=None, alias='HouseStyle')
    roof_style:      Optional[str] = Field(default=None, alias='RoofStyle')
    roof_matl:       Optional[str] = Field(default=None, alias='RoofMatl')
    exterior1st:     Optional[str] = Field(default=None, alias='Exterior1st')
    exterior2nd:     Optional[str] = Field(default=None, alias='Exterior2nd')
    mas_vnr_type:    Optional[str] = Field(default=None, alias='MasVnrType')
    exter_qual:      Optional[str] = Field(default=None, alias='ExterQual')
    exter_cond:      Optional[str] = Field(default=None, alias='ExterCond')
    foundation:      Optional[str] = Field(default=None, alias='Foundation')
    bsmt_qual:       Optional[str] = Field(default=None, alias='BsmtQual')
    bsmt_cond:       Optional[str] = Field(default=None, alias='BsmtCond')
    bsmt_exposure:   Optional[str] = Field(default=None, alias='BsmtExposure')
    bsmt_fin_type1:  Optional[str] = Field(default=None, alias='BsmtFinType1')
    bsmt_fin_type2:  Optional[str] = Field(default=None, alias='BsmtFinType2')
    heating:         Optional[str] = Field(default=None, alias='Heating')
    heating_qc:      Optional[str] = Field(default=None, alias='HeatingQC')
    central_air:     Optional[str] = Field(default=None, alias='CentralAir')
    electrical:      Optional[str] = Field(default=None, alias='Electrical')
    kitchen_qual:    Optional[str] = Field(default=None, alias='KitchenQual')
    functional:      Optional[str] = Field(default=None, alias='Functional')
    fireplace_qu:    Optional[str] = Field(default=None, alias='FireplaceQu')
    garage_type:     Optional[str] = Field(default=None, alias='GarageType')
    garage_finish:   Optional[str] = Field(default=None, alias='GarageFinish')
    garage_qual:     Optional[str] = Field(default=None, alias='GarageQual')
    garage_cond:     Optional[str] = Field(default=None, alias='GarageCond')
    paved_drive:     Optional[str] = Field(default=None, alias='PavedDrive')
    pool_qc:         Optional[str] = Field(default=None, alias='PoolQC')
    fence:           Optional[str] = Field(default=None, alias='Fence')
    misc_feature:    Optional[str] = Field(default=None, alias='MiscFeature')
    sale_type:       Optional[str] = Field(default=None, alias='SaleType')
    sale_condition:  Optional[str] = Field(default=None, alias='SaleCondition')

class HouseOutput(BaseModel):
    SalePrice: float
    status: str = "success"

@app.post("/predict/")
async def predict_price(input_data: HouseInput):
    try:
        data_dict = input_data.model_dump(by_alias=True)
        
        price = predict.get_prediction(data_dict)
        return {"SalePrice": price, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))