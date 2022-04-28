#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import glob
import pandas as pd


def map_masking(map_arr, mask_arr, fill_val=np.nan, real_val=0):
    """
    Masking function to replace bad values (>0) of a 2d np.ndarray with a fill value

    Code:
        map_arr[mask_arr != real_val] = fill_val


    Parameters:
        map_arr: ndarray
            2D ndarray with original values

        mask_arr: ndarray
            2D ndarray of the mask array and must be the same shape as map_arr

        fill_val: int, float, np,nan, np.inf (Default=np.nan)
            The value that will be substituted for the bad values in the map array

        real_val: int or float (Default=0)
            The value that repersents good value in the mask array
    """

    map_arr[mask_arr != real_val] = fill_val

    return map_arr


def balmer_dec(obs_haflux, obs_hbflux):
    """
    Calculates Balmer decrement (Ha_flux / Hb_flux)

    Paremeters:
        obs_haflux: float, numpy.ndarray, marvin.tools.quantities.map.Map
            Observed Halpha flux value or Map

        obs_hbflux: float, numpy.ndarray, marvin.tools.quantities.map.Map
            Observed Hbeta flux value or Map
    """
    bdec = obs_haflux / obs_hbflux
    return bdec


def c00_k(wavelength, Rv=4.05):
    """
    Calzetti extinction curve 2000 (Good for starburst like dust)
    Paremeters:

        wavelength: float,
            units of microns

        Rv: float,
            normilaztion of the extinction curve

    Return:
        k_lambda : float
            Extinction curve (k) for given wavelength
    """
    if np.logical_and(wavelength >= 0.63, wavelength <= 2.2):
        # k_lambda = 1.17 * (-1.857 + (1.040/rest_lam)) + 1.78 # Calzetti 2001 obscuration code
        k_lambda = 2.659 * (-1.857 + (1.040 / wavelength)) + Rv  # Calzetti 2000
        # print('k_lambda={}'.format(k_lambda))
    elif np.logical_and(wavelength >= 0.12, wavelength < 0.63):
        # k_lambda = 1.17 * (-2.156 + (1.509/rest_lam) - (0.198/rest_lam**2) + (0.011/rest_lam**3)) + 1.78 # Calzetti 2001 obscuration code
        k_lambda = (
            2.659
            * (
                -2.156
                + (1.509 / wavelength)
                - (0.198 / wavelength**2)
                + (0.011 / wavelength**3)
            )
            + Rv
        )
        # print('k_lambda={}'.format(k_lambda))
    else:
        pass
        # print(
        #     "Rest wavelength is not in range or not in Angstroms {}".format(wavelength)
        # )

    return k_lambda


def c00_kcorr(obs_flux, obs_wavelength, bdec, Rv=4.05):
    """
    Extinction correction using the Calzetti extinction curve

    Paremeter:
        obs_flux: float, numpy.ndarray, marvin.tools.quantities.map.Map
            Obsereved flux value or map [1e-17erg/cm^2/s/spaxel]

        obs_wavelength: float,
            wavelength [micron]

        bdec: float, numpy.ndarray, marvin.tools.quantities.map.Map
            Balmer Decrement (Ha/Hb)
            *Must be same type as obs_flux*

        Rv: float, curve normilization

    Return:
        int_flux: float
            intrinsic flux 1e-17*erg/cm^2/s


    """

    # Observe flux
    # print('Observe Flux = {} 1e-17erg/cm^2/spaxel/s'.format(obs_flux))

    # Observe extinction curve
    obs_k = c00_k(obs_wavelength, Rv=Rv)
    # print('k({} mircon) = {}'.format(obs_wavelength, obs_k))

    # Ha, Hb extinction curve
    k_Ha = c00_k(0.6564, Rv)
    # print('k({} mircon) = {}'.format(.6564, k_Ha))
    k_Hb = c00_k(0.4864, Rv)
    # print('k({} mircon) = {}'.format(.4864, k_Hb))

    # Optical depth tau
    tau = np.log10(bdec / 2.86)  # possibly np.log10
    # print('Balmer optical depth={}'.format(tau))

    # color excess(gas) - Battisti et al 2017 eqn 2
    EBV_gas = (1.086 / (k_Hb - k_Ha)) * tau
    EBV_star = EBV_gas * 0.44
    # print('E(B-V)gas={}'.format(EBV_gas))
    # print('E(B-V)star={}'.format(EBV_star))

    # Intrinsic flux [erg/cm^2/s]
    int_flux = obs_flux * 10 ** (0.4 * obs_k * EBV_star)
    # print('intrinsic_flux = {} 1e-17erg/cm^2/s'.format(int_flux))
    # print('')

    return int_flux


def sfr_ha_map(halphadc_map, z, bin_area_map):
    """
    Paremters:
        halphadc_map

    """
    # sfr_map = (7.9e-42 * halphadc_map) / 1.53

    # Calculate Luminosity distance
    cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
    lum_d_mpc = cosmo.luminosity_distance(z)
    lum_d_cm = lum_d_mpc.to(u.cm).value  # convert from Mpc to cm
    # print('Lum Distance {} converted to {} cm'.format(lum_d_mpc, lum_d_cm))

    # Convert flux to Luminosity
    # Halpha Flux : 1e-17*erg/s/cm^2/spaxel
    # From Hoggs 2000 paper
    lum_ha_map = halphadc_map * (1 + z) * (4 * np.pi * (lum_d_cm**2))
    # print('Halpha Flux: ',halphadc_map[27,27])
    # print('Halpha Luminosity: ',l_ha[27,27])

    # sfr - Salpeter initial mass function Hao et al. (2011)
    sfr_map = 8.79e-42 * lum_ha_map  # [Msolar/yr]
    # print('SFR: ',sfr_map[27,27])

    # Estimate the spaxel diameter in kpc
    spaxel_diamter_in_kpc = cosmo.kpc_proper_per_arcmin(z).to(
        u.kpc / u.arcsec
    )  # [kpc/arcsec]
    spaxel_area_in_kpc2 = (
        spaxel_diamter_in_kpc**2
    )  # [kpc2/arcsec2] Since my data is binned the area is a roughly a box for each pixel

    # Using the Bin Area to calculate the Area of each bin in Kpc
    spaxel_area_kpc2 = spaxel_area_in_kpc2 * bin_area_map
    # Bin length from the Bin Area Map - np.sqrt(BIN_AREA[arcsec^2])
    # bin_length = np.sqrt(bin_area_map) # [arcsec]

    # scale = 1 / 206265 * D * 1e6  # 1 radian = 206265 arcsec [pc / arcsec]
    # spaxel_area_pc2 = (scale * spaxel_size) ** 2 * u.pc ** 2  # [pc^2]
    # spaxel_area_kpc2 = spaxel_area_pc2.to(u.kpc ** 2)

    # Calculate the SFRD of each spaxel
    sfrd_map_kpc2 = sfr_map / spaxel_area_kpc2.value  # [Msolar/yr/kpc2]

    ##Measeure the diameter of each spaxel with respect to the galaxy distance
    # spaxel_diamter_in_kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec) * (0.5 * u.arcsec)
    ##print('Diamter of each spaxel (0.5 arcsec) on the galaxy: {}'.format(spaxel_diamter_in_kpc))
    #
    ##Area of each spaxel
    # Area_of_each_spaxels =  4 * np.pi * (spaxel_diamter_in_kpc/2)**2
    ##print('Area of each spaxel: {}'.format(Area_of_each_spaxels))
    #
    ##Calculate the SFRD of each spaxel
    # sfrd_map = sfr_map / Area_of_each_spaxels
    ##print('SFRD: ',sfrd_map[27,27])

    return (
        sfr_map,
        lum_ha_map,
        sfrd_map_kpc2,
        spaxel_area_kpc2.value,
        spaxel_diamter_in_kpc.value,
    )


# def radius_ratio_map(plateifu):
#     """
#     platifu: str

#     """
#     # Galaxy effective radius
#     r_eff = float(Maps(plateifu=plateifu).header["reff"])  # arcsec
#     # print('Effective Radius', Reff)

#     # Galaxy elliptical radius
#     r_map = Maps(plateifu=plateifu).spx_ellcoo_elliptical_radius.value  # arcsec
#     # print('Elliptical Radius', R)

#     # R/Reff
#     radius_ratio = r_map / r_eff

#     return r_map, radius_ratio, r_eff


def n2o2_Z_map(nii6585_fluxmap, oii3727_fluxmap, oii3729_fluxmap):
    # Kewley and Dopita - 2002 - Using Strong Lines to Estimate Abundances in Extra (KD02)
    # NII/OII diagnostic [N II] λ6584/[O II] λ3727,3729
    # Note: Independent of ionization parameter, Z>=8.6 for a reliable abundance
    fratio_map = np.log10(nii6585_fluxmap / (oii3727_fluxmap + oii3729_fluxmap))
    Z_map = (
        np.log10(1.54020 + (1.26602 * fratio_map) + (0.167977 * fratio_map**2)) + 8.93
    )  # KD02 (eq 5&7) [Z = log(O/H) +12]
    return fratio_map, Z_map


def o3n2_metal_map(o3_5008_fluxmap, nii6585_fluxmap, ha_fluxmap, hb_fluxmap):
    # Marino et al. 2013 O3N2 diagnostic
    # O3N2 diagnostic [OIII]λ5007/Hbeta * Halpha/[NII]λ6583
    fratio_map = np.log10(
        (o3_5008_fluxmap / hb_fluxmap) * (ha_fluxmap / nii6585_fluxmap)
    )
    Z_map = 8.505 - 0.221 * fratio_map
    return fratio_map, Z_map


def n2_metal_map(nii6585_fluxmap, ha_fluxmap):
    # Marino et al. 2013 N2 diagnostic
    # N2 diagnostic log([NII]λ6583/Halpha)
    fratio_map = np.log10(nii6585_fluxmap / ha_fluxmap)
    Z_map = 8.667 - 0.455 * fratio_map  # Log[O/H] + 12
    return fratio_map, Z_map


# def n2s2_metal_map(nii6585_fluxmap, sii6718_fluxmap, sii6732_fluxmap, ha_fluxmap):
#    # Marino et al. 2013 N2 diagnostic
#    # N2S2 diagnostic log([NII]λ6583/Halpha)
#    n2_ratio_map, n2_Z_map = n2_metal_map(nii6585_fluxmap, ha_fluxmap) #
#    fratio_map = np.log10(nii6585_fluxmap / (sii6718_fluxmap + sii6732_fluxmap))
#    metal_map = 8.77 + fratio_map + 0.264 * n2_Z_map
#    return ratio_map, metal_map


# Pipe3D Maps
def pipe3d_maps(plateifu, sample="bbrd"):
    """
    Parameters:
        plateifu: str
            MaNGA plateifu

    Return:
    age_l, age_Z, age_err, stel_vel, vel_err, v_disp, v_err, ml_ratio, mass_density
    """
    # SSP
    hdu = fits.open(
        "/Users/mmckay/Desktop/research/FMR_MZR/{}_pipe3d_fits/manga-{}.Pipe3D.cube.fits".format(
            sample, plateifu
        )
    )
    age_l = hdu[1].data[5, :, :]  # Gyr - Luminosity Weighted age
    age_m = hdu[1].data[6, :, :]  # Gyr - Mass Weighted age
    age_err = hdu[1].data[7, :, :]  # Gyr - Error of the age
    # Luminosity Weighted metallicity of the stellar population (where Z=0.02 is solar metallicity)
    metal_l = hdu[1].data[8, :, :]

    metal_m = hdu[1].data[
        9, :, :
    ]  # Mass Weighted metallicity of the stellar population
    metal_err = hdu[1].data[
        10, :, :
    ]  # Error of the metallicity of the stellar population
    stel_vel = hdu[1].data[13, :, :]  # [km/s] Velocity map of the stellar population
    vel_err = hdu[1].data[14, :, :]  # Error in the velocity of the stellar population
    v_disp = hdu[1].data[
        15, :, :
    ]  # Velocity dispersion of the stellar population (sigma)
    v_err = hdu[1].data[
        16, :, :
    ]  # Error in velocity dispersion of the stellar population
    ml_ratio = hdu[1].data[
        17, :, :
    ]  # [Log(Msun/Lsun)] Average mass-to-light ratio of the stellar population
    mass_rho = hdu[1].data[
        19, :, :
    ]  # [Log(Msun/spaxels^2)] Stellar Mass density per pixel with dust correction
    mass_rho_err = hdu[1].data[
        20, :, :
    ]  # [Log(Msun/spaxels^2)] Stellar Mass density per pixel with dust correction

    # INDCICES - not corrected for velocity dispersion
    d4000_index = hdu[4].data[5, :, :]  # D4000 index map
    d4000_err = hdu[4].data[13, :, :]  # D4000 index err map
    hdelta_index = hdu[4].data[6, :, :]  # Hdelta index map
    hdelta_err = hdu[4].data[14, :, :]  # Hdelta index map

    hdu.close()

    return (
        age_l,
        age_m,
        age_err,
        metal_l,
        metal_m,
        metal_err,
        stel_vel,
        vel_err,
        v_disp,
        v_err,
        ml_ratio,
        mass_rho,
        mass_rho_err,
        d4000_index,
        d4000_err,
        hdelta_index,
        hdelta_err,
    )


# Resolved BPT Classification
# def bpt_ifu_classification(o3_5008_mapdc, hb_mapdc, nii6585_mapdc, ha_mapdc):
#     # Classify spaxel using BPT from Kewley et al 2006?
#     #
#     # Compute BPT ratios
#     # OIII/Hbeta, NII/halpha extinction corrected
#     o3hb_ratio_orig = np.log10(o3_5008_mapdc / hb_mapdc)
#     n2ha_ratio_orig = np.log10(nii6585_mapdc / ha_mapdc)

#     o3hb_ratio = np.copy(o3hb_ratio_orig)
#     n2ha_ratio = np.copy(n2ha_ratio_orig)

#     ## OIII/Hbeta, NII/halpha non-extinction corrected
#     # o3hb_ratio_ne = np.log10(o3_5008_val / hb_val)
#     # n2ha_ratio_ne = np.log10(nii6585_val / ha_val)

#     # Classify spaxels
#     agn_mask = (o3hb_ratio > (0.61 / (n2ha_ratio - 0.47)) + 1.19) | (
#         n2ha_ratio > 0.4
#     )  # AGN class
#     comp_mask = (o3hb_ratio <= (0.61 / (n2ha_ratio - 0.47)) + 1.19) & (
#         o3hb_ratio >= (0.61 / (n2ha_ratio - 0.05)) + 1.3
#     )  # Composite class
#     sf_mask = (
#         (o3hb_ratio <= (0.61 / (n2ha_ratio - 0.05)) + 1.3)
#         & (o3hb_ratio <= (0.61 / (n2ha_ratio - 0.47)) + 1.19)
#         & (n2ha_ratio <= 0.4)
#     )  # SF class

#     # Combine mask to create BPT image
#     # Where 1:SF, 2:Comp, 3:AGN
#     sf_mask_num = np.where(sf_mask == True, 1, sf_mask)
#     comp_mask_num = np.where(comp_mask == True, 2, comp_mask)
#     agn_mask_num = np.where(agn_mask == True, 3, agn_mask)
#     combo_mask_num = agn_mask_num + sf_mask_num + comp_mask_num

#     return o3hb_ratio_orig, n2ha_ratio_orig, combo_mask_num


def bpt_o3hb_n2ha_diagnostic_polyfit(
    n2ha_sf_range=np.linspace(-2.0, 0, 200), n2ha_comp_range=np.linspace(-2.0, 0.4, 200)
):
    ka03_o3ha_sf_fit = (0.61 / (n2ha_sf_range - 0.05)) + 1.3
    ke06_o3ha_comp_fit = (0.61 / (n2ha_comp_range - 0.47)) + 1.19
    return ka03_o3ha_sf_fit, ke06_o3ha_comp_fit


def bpt_n2ha_2dmap(o3_5008_fluxmap, hb_fluxmap, nii6585_fluxmap, ha_fluxmap):

    # Compute Ratios
    o3hb_ratio_map_orig = o3_5008_fluxmap / hb_fluxmap
    n2ha_ratio_map_orig = nii6585_fluxmap / ha_fluxmap

    # Make copy of ratio maps
    o3hb_ratio_map = np.log10(np.copy(o3hb_ratio_map_orig))
    n2ha_ratio_map = np.log10(np.copy(n2ha_ratio_map_orig))

    # Calculate the estimated value for OIII/Hbeta map from NII/Halpha
    ka03_o3ha_sf_fit, ke06_o3ha_comp_fit = bpt_o3hb_n2ha_diagnostic_polyfit(
        n2ha_ratio_map, n2ha_ratio_map
    )

    # Select Log(OIII/Hbeta) values below the sf polyfit curve
    o3ha_sf_spaxels = np.copy(o3hb_ratio_map)
    o3ha_sf_spaxels[o3ha_sf_spaxels >= ka03_o3ha_sf_fit] = np.nan
    o3ha_sf_spaxels[n2ha_ratio_map >= -0.1] = np.nan

    # Select Log(OIII/Hbeta) values above the sf polyfit curve and below the composite curves
    o3ha_comp_spaxels = np.copy(o3hb_ratio_map)
    o3ha_comp_spaxels[
        o3ha_comp_spaxels == o3ha_sf_spaxels
    ] = np.nan  # set sf classifed spaxel to nan
    o3ha_comp_spaxels[o3ha_comp_spaxels >= ke06_o3ha_comp_fit] = np.nan
    o3ha_comp_spaxels[n2ha_ratio_map >= 0.25] = np.nan

    # Select Log(OIII/Hbeta) values above the comp ployfit
    o3ha_agn_spaxels = np.copy(o3hb_ratio_map)
    o3ha_agn_spaxels[
        o3ha_agn_spaxels == o3ha_sf_spaxels
    ] = np.nan  # set sf classifed spaxel to nan
    o3ha_agn_spaxels[
        o3ha_agn_spaxels == o3ha_comp_spaxels
    ] = np.nan  # set sf classifed spaxel to nan

    # SF spaxel constant map value=1
    sf_spax_constmap = np.copy(o3ha_sf_spaxels)
    sf_spax_constmap[o3ha_sf_spaxels > -99.0] = 1.0

    # Composite spaxel constant map value=2
    comp_spax_constmap = np.copy(o3ha_comp_spaxels)
    comp_spax_constmap[o3ha_comp_spaxels > -99.0] = 2.0

    # Composite spaxel constant map value=2
    agn_spax_constmap = np.copy(o3ha_agn_spaxels)
    agn_spax_constmap[o3ha_agn_spaxels > -99.0] = 3.0

    # BPT datacube
    bpt_3d_constant_map = np.dstack(
        (sf_spax_constmap, comp_spax_constmap, agn_spax_constmap)
    )

    # Resolved BPT 2D map
    bpt_2d_constant_map = np.nansum(bpt_3d_constant_map, axis=2)
    bpt_2d_constant_map[bpt_2d_constant_map == 0.0] = np.nan

    return o3hb_ratio_map, n2ha_ratio_map, bpt_2d_constant_map


def write_maps2fits_mpldap(plateifu="None", mode="local", sample="bbrd", z="False"):

    # Calculate SNR for MaNGA maps  using the eqaution found in marvin.tools source code cite later

    # Read in MPL-11 DAP FITS file from local directory
    dap_gal_fits = fits.open(
        "/Volumes/lil_onyx/{}_dapfits/manga-{}-MAPS-HYB10-MILESHC-MASTARSSP.fits".format(
            sample, plateifu
        )
    )

    # Read in MaNGA Maps from local directory with MPL11 DAP files -
    # Emmison line units [1e-17 erg / (cm s spaxel)]
    ha_6564_map = dap_gal_fits["EMLINE_GFLUX"].data[23]
    hb_4862_map = dap_gal_fits["EMLINE_GFLUX"].data[14]
    nii6585_map = dap_gal_fits["EMLINE_GFLUX"].data[24]
    oii3727_map = dap_gal_fits["EMLINE_GFLUX"].data[0]
    oii3729_map = dap_gal_fits["EMLINE_GFLUX"].data[1]
    o3_5008_map = dap_gal_fits["EMLINE_GFLUX"].data[16]
    sii6718_map = dap_gal_fits["EMLINE_GFLUX"].data[25]
    sii6732_map = dap_gal_fits["EMLINE_GFLUX"].data[26]

    # Data Quality Maps
    ha_6564_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[23]
    hb_4862_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[14]
    nii6585_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[24]
    oii3727_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[0]
    oii3729_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[1]
    o3_5008_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[16]
    sii6718_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[25]
    sii6732_dqmap = dap_gal_fits["EMLINE_GFLUX_MASK"].data[26]

    # SNR Maps [return numpy.abs(self.value * numpy.sqrt(self.ivar)) - Marvin Software]
    ha_6564_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[23]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[23])
    )
    hb_4862_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[15]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[15])
    )
    nii6585_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[23]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[23])
    )
    oii3727_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[1]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[1])
    )
    oii3729_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[2]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[2])
    )
    o3_5008_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[17]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[17])
    )
    sii6718_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[26]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[26])
    )
    sii6732_snr_map = np.abs(
        dap_gal_fits["EMLINE_GFLUX"].data[27]
        * np.sqrt(dap_gal_fits["EMLINE_GFLUX_IVAR"].data[27])
    )

    # Convert Inverse Varience maps to standard error maps
    ha_6564_error_map = np.sqrt(1 / (dap_gal_fits[31].data[23]))
    hb_4862_error_map = np.sqrt(1 / (dap_gal_fits[31].data[14]))
    nii6585_error_map = np.sqrt(1 / (dap_gal_fits[31].data[24]))
    oii3727_error_map = np.sqrt(1 / (dap_gal_fits[31].data[0]))
    oii3729_error_map = np.sqrt(1 / (dap_gal_fits[31].data[1]))
    o3_5008_error_map = np.sqrt(1 / (dap_gal_fits[31].data[16]))
    sii6718_error_map = np.sqrt(1 / (dap_gal_fits[31].data[25]))
    sii6732_error_map = np.sqrt(1 / (dap_gal_fits[31].data[26]))

    # apply bad pixels mask to Maps with a fill val = np.nan
    ha_map_clean = map_masking(
        map_arr=ha_6564_map, mask_arr=ha_6564_dqmap, fill_val=np.nan, real_val=0
    )  # remove bad spaxels
    hb_map_clean = map_masking(
        map_arr=hb_4862_map, mask_arr=hb_4862_dqmap, fill_val=np.nan, real_val=0
    )
    nii6585_map_clean = map_masking(
        map_arr=nii6585_map, mask_arr=nii6585_dqmap, fill_val=np.nan, real_val=0
    )
    oii3727_map_clean = map_masking(
        map_arr=oii3727_map, mask_arr=oii3727_dqmap, fill_val=np.nan, real_val=0
    )
    oii3729_map_clean = map_masking(
        map_arr=oii3729_map, mask_arr=oii3729_dqmap, fill_val=np.nan, real_val=0
    )
    o3_5008_map_clean = map_masking(
        map_arr=o3_5008_map, mask_arr=o3_5008_dqmap, fill_val=np.nan, real_val=0
    )
    sii6718_map_clean = map_masking(
        map_arr=sii6718_map, mask_arr=sii6718_dqmap, fill_val=np.nan, real_val=0
    )
    sii6732_map_clean = map_masking(
        map_arr=sii6732_map, mask_arr=sii6732_dqmap, fill_val=np.nan, real_val=0
    )

    # Balmer Decrrement
    bdec_map = balmer_dec(obs_haflux=ha_map_clean, obs_hbflux=hb_map_clean)
    bdec_map[
        bdec_map == np.inf
    ] = 0.0  # Replace inf values with zero incase hbeta map had zero values

    # Write Maps to FITS file
    # Writes a new fits for the data
    new_hdul = fits.HDUList()
    # Flux map with masked spaxels removed
    new_hdul.append(
        fits.ImageHDU(ha_map_clean, ver=1, name="Halpha")
    )  # Halpha [1e-17 erg / (cm2 s spaxel)]

    # Run kcorrection function on emission lines
    ha_mapdc = c00_kcorr(ha_map_clean, 0.6564, bdec_map, Rv=4.05)
    hb_mapdc = c00_kcorr(hb_map_clean, 0.4864, bdec_map, Rv=4.05)
    nii6585_mapdc = c00_kcorr(nii6585_map_clean, 0.6585, bdec_map, Rv=4.05)
    oii3727_mapdc = c00_kcorr(oii3727_map_clean, 0.3727, bdec_map, Rv=4.05)
    oii3729_mapdc = c00_kcorr(oii3729_map_clean, 0.3729, bdec_map, Rv=4.05)
    o3_5008_mapdc = c00_kcorr(o3_5008_map_clean, 0.5008, bdec_map, Rv=4.05)
    sii6718_mapdc = c00_kcorr(sii6718_map_clean, 0.6718, bdec_map, Rv=4.05)
    sii6732_mapdc = c00_kcorr(sii6732_map_clean, 0.6732, bdec_map, Rv=4.05)

    # Add k-corrected emission line fluxes to FITS file - not logged
    new_hdul.append(
        fits.ImageHDU(ha_mapdc, name="Halpha kcorr", ver=2)
    )  # Halpha k-corrected [1e-17 erg / (cm2 s spaxel)]
    new_hdul.append(fits.ImageHDU(hb_mapdc, name="Hbeta kcorr", ver=3))
    new_hdul.append(fits.ImageHDU(nii6585_mapdc, name="NII6585 kcorr", ver=4))
    new_hdul.append(fits.ImageHDU(oii3727_mapdc, name="OII3727 kcorr", ver=5))
    new_hdul.append(fits.ImageHDU(oii3729_mapdc, name="OII3729 kcorr", ver=6))
    new_hdul.append(fits.ImageHDU(o3_5008_mapdc, name="OIII5008 kcorr", ver=7))
    new_hdul.append(fits.ImageHDU(sii6718_mapdc, name="SII6718 kcorr", ver=8))
    new_hdul.append(fits.ImageHDU(sii6732_mapdc, name="SII6732 kcorr", ver=9))

    # S/N ratio Maps - SNR calculation
    new_hdul.append(fits.ImageHDU(ha_6564_snr_map, name="Halpha SNR", ver=10))  # SNR
    new_hdul.append(fits.ImageHDU(hb_4862_snr_map, name="Hbeta SNR", ver=11))  # SNR
    new_hdul.append(fits.ImageHDU(nii6585_snr_map, name="NII6585 SNR", ver=12))  # SNR
    new_hdul.append(fits.ImageHDU(oii3727_snr_map, name="OII3727 SNR", ver=13))  # SNR
    new_hdul.append(fits.ImageHDU(oii3729_snr_map, name="OII3729 SNR", ver=14))  # SNR
    new_hdul.append(fits.ImageHDU(o3_5008_snr_map, name="OIII5008 SNR", ver=15))  # SNR
    new_hdul.append(fits.ImageHDU(sii6718_snr_map, name="SII6718 SNR", ver=16))  # SNR
    new_hdul.append(fits.ImageHDU(sii6732_snr_map, name="SII6732 SNR", ver=17))  # SNR

    # Emission line flux error [1-17 erg / s /spaxel / cm2]
    new_hdul.append(fits.ImageHDU(ha_6564_error_map, name="Halpha ERROR", ver=18))
    new_hdul.append(fits.ImageHDU(hb_4862_error_map, name="Hbeta ERROR", ver=19))
    new_hdul.append(fits.ImageHDU(nii6585_error_map, name="NII6585 ERROR", ver=20))
    new_hdul.append(fits.ImageHDU(oii3727_error_map, name="OII3727 ERROR", ver=21))
    new_hdul.append(fits.ImageHDU(oii3729_error_map, name="OII3729 ERROR", ver=22))
    new_hdul.append(fits.ImageHDU(o3_5008_error_map, name="OIII5008 ERROR", ver=23))
    new_hdul.append(fits.ImageHDU(sii6718_error_map, name="SII6718 ERROR", ver=24))
    new_hdul.append(fits.ImageHDU(sii6732_error_map, name="SII6732 ERROR", ver=25))

    # g-band SNR per pixel map
    # mean g-band weighted SNR per pixel
    # snr_map = dap_gal_fits[5].data
    new_hdul.append(fits.ImageHDU(ha_6564_dqmap, ver=26, name="HALPHA_DQ"))

    # My Caluclation for SFRD
    bin_area_map = dap_gal_fits[9].data
    sfr_map, l_ha, sfrd_map, spaxel_area_kpc, spaxel_diamter_in_kpc = sfr_ha_map(
        halphadc_map=(ha_mapdc / 1e17), z=z, bin_area_map=bin_area_map
    )

    # Add Star formation
    new_hdul.append(fits.ImageHDU(bdec_map, ver=27, name="Ha/Hb"))  # Balmer decerment
    new_hdul.append(fits.ImageHDU(l_ha, ver=28, name="Ha_Lum"))  # erg / (s spaxel)
    new_hdul.append(
        fits.ImageHDU(np.log10(sfr_map), ver=29, name="logSFR")
    )  # Msolar/yr
    new_hdul.append(
        fits.ImageHDU(np.log10(sfrd_map), ver=30, name="logSFR Density")
    )  # Msolar/yr/kpc^2

    # Radius Map
    r_map = dap_gal_fits[2].data[0]  # arcsec
    radius_ratio = dap_gal_fits[32].data[24]
    try:
        r_eff = dap_gal_fits[0].header["REFF"]
    except:
        r_eff = -9999  # place holder value

    # Add Radius Maps
    new_hdul.append(fits.ImageHDU(r_map, ver=31, name="Ellip R"))  # arcsec
    new_hdul.append(fits.ImageHDU(r_map / r_eff, ver=32, name="R/Reff"))

    # Metallicity log(O/H)+12
    # Log(O/H)+12 - N2O2
    n2o2_ratiomap, n2o2_metalmap = n2o2_Z_map(
        nii6585_mapdc, oii3727_mapdc, oii3729_mapdc
    )

    # Log(O/H)+12 - O3N2
    o3n2_ratiomap, o3n2_metalmap = o3n2_metal_map(
        o3_5008_fluxmap=o3_5008_mapdc,
        nii6585_fluxmap=nii6585_mapdc,
        ha_fluxmap=ha_mapdc,
        hb_fluxmap=hb_mapdc,
    )

    # Log(O/H)+12 - NII/Ha
    n2_ratiomap, n2_metalmap = n2_metal_map(
        nii6585_fluxmap=nii6585_mapdc, ha_fluxmap=ha_mapdc
    )
    # Add Metallicity Maps
    new_hdul.append(fits.ImageHDU(n2o2_metalmap, ver=33, name="Log(O/H)+12_[N2O2]"))
    new_hdul.append(fits.ImageHDU(o3n2_metalmap, ver=34, name="Log(O/H)+12_[O3N2]"))
    new_hdul.append(fits.ImageHDU(n2_metalmap, ver=35, name="Log(O/H)+12_[N2]"))

    # Pipe3D Maps
    # Pipe3D Maps - Remove
    # age_l, age_m, age_err, metal_l, metal_m, metal_err, stel_vel,
    # vel_err, v_disp, v_err, ml_ratio, mass_rho, d4000_index, d4000_err,
    # hdelta_index, hdelta_err = pipe3d_maps(plateifu, sample=sample)

    # Read in FITS files Pipe3D
    pipe3d_gal_fits = fits.open(
        "/Volumes/lil_onyx/{}_Pipe3Dfits/manga-{}.Pipe3D.cube.fits".format(
            sample, plateifu
        )
    )
    age_l = pipe3d_gal_fits["SSP"].data[5]
    age_m = pipe3d_gal_fits["SSP"].data[6]
    age_err = pipe3d_gal_fits["SSP"].data[7]
    metal_l = pipe3d_gal_fits["SSP"].data[8]
    metal_m = pipe3d_gal_fits["SSP"].data[9]
    metal_err = pipe3d_gal_fits["SSP"].data[10]
    stel_vel = pipe3d_gal_fits["SSP"].data[13]
    vel_err = pipe3d_gal_fits["SSP"].data[14]
    v_disp = pipe3d_gal_fits["SSP"].data[15]
    v_err = pipe3d_gal_fits["SSP"].data[16]
    ml_ratio = pipe3d_gal_fits["SSP"].data[17]
    mass_rho = pipe3d_gal_fits["SSP"].data[
        19
    ]  # 18 is not dust corrected - 19 is dust corrected
    mass_rho_err = pipe3d_gal_fits[1].data[20]

    # Indexs are not corrected for velocity dispersion
    d4000_index = pipe3d_gal_fits[3].data[5]
    d4000_err = pipe3d_gal_fits[3].data[13]
    hdelta_index = pipe3d_gal_fits[3].data[0]
    hdelta_err = pipe3d_gal_fits[3].data[8]

    # Add Pipe3D maps to FITS
    new_hdul.append(fits.ImageHDU(age_l, ver=36, name="Gyr_lw"))
    new_hdul.append(fits.ImageHDU(age_m, ver=37, name="Gyr_mw"))
    new_hdul.append(fits.ImageHDU(age_err, ver=38, name="Gyr_err"))
    new_hdul.append(fits.ImageHDU(metal_l, ver=39, name="SP_ZsubL"))
    new_hdul.append(fits.ImageHDU(metal_m, ver=40, name="SP_ZsubM"))
    new_hdul.append(fits.ImageHDU(metal_err, ver=41, name="SP_Zerr"))
    new_hdul.append(fits.ImageHDU(stel_vel, ver=42, name="Vel_km/s"))
    new_hdul.append(fits.ImageHDU(vel_err, ver=43, name="Vel_err]"))
    new_hdul.append(fits.ImageHDU(v_disp, ver=44, name="Vdisp_km/s"))
    new_hdul.append(fits.ImageHDU(v_err, ver=45, name="Vdisp_err"))
    new_hdul.append(fits.ImageHDU(ml_ratio, ver=46, name="M/L"))
    new_hdul.append(
        fits.ImageHDU(mass_rho, ver=47, name="Msun/spx2")
    )  # Dust corrected mass density
    new_hdul.append(fits.ImageHDU(mass_rho / 0.25, ver=48, name="Msun/arcs2"))
    new_hdul.append(fits.ImageHDU(mass_rho_err, ver=49, name="Msun_ERR"))
    new_hdul.append(fits.ImageHDU(d4000_index, ver=50, name="D4000"))
    new_hdul.append(fits.ImageHDU(d4000_err, ver=51, name="D4000_err"))
    new_hdul.append(fits.ImageHDU(hdelta_index, ver=52, name="Hdelta"))
    new_hdul.append(fits.ImageHDU(hdelta_err, ver=53, name="Hdelta_err"))

    # BPT classification mask
    # o3hb_ratio, n2ha_ratio, combo_mask_num = bpt_ifu_classification(
    #     o3_5008_mapdc, hb_mapdc, nii6585_mapdc, ha_mapdc
    # )
    o3hb_ratio_map, n2ha_ratio_map, bpt_2d_constant_map = bpt_n2ha_2dmap(
        o3_5008_map_clean, hb_map_clean, nii6585_map_clean, ha_map_clean
    )
    new_hdul.append(fits.ImageHDU(o3hb_ratio_map, ver=54, name="O3HB_RATIO"))
    new_hdul.append(fits.ImageHDU(n2ha_ratio_map, ver=55, name="N2HA_RATIO"))
    new_hdul.append(fits.ImageHDU(bpt_2d_constant_map, ver=56, name="BPT_CLASS"))

    # Bluck et al. 2020 delta SFR = SFRD - Bluck least square minimization
    sfms_fit_map = 0.90 * mass_rho - 9.57  # Bluck least square minimization fit
    sfms_fit_map_upperlimit = 1.21 * mass_rho - 11.5  # upper limit
    sfms_fit_map_lowerlimit = 0.68 * mass_rho - 7.64  # lower limit
    delta_sfr = np.log10(sfrd_map) - sfms_fit_map  # Delta SFR

    new_hdul.append(fits.ImageHDU(delta_sfr, ver=57, name="B20_DELTASFR"))
    new_hdul.append(
        fits.ImageHDU(np.log10(sfr_map) - mass_rho, ver=58, name="Log_Sigma_sSFR")
    )
    new_hdul.append(
        fits.ImageHDU(sfms_fit_map_upperlimit, ver=59, name="B20_SFRD_UPLIM")
    )
    new_hdul.append(
        fits.ImageHDU(sfms_fit_map_lowerlimit, ver=60, name="B20_SFRD_LOWLIM")
    )

    # D4000 and Dn4000 from DAP Maps (Must be corrected to convert measurements to 0 velocity dispersion at the MILES resolution)
    # dap_d4000_map = dap_gal_fits["SPECINDEX"].data[43]
    # dap_dn4000_map = dap_gal_fits["SPECINDEX"].data[44]
    # dap_hdelta_A_map = dap_gal_fits["SPECINDEX"].data[21]
    # dap_hdelta_F_map = dap_gal_fits["SPECINDEX"].data[23]

    # Remove bad DQ spaxel_size
    dap_d4000_map_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX"].data[43],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[43],
        fill_val=np.nan,
        real_val=0,
    )

    dap_dn4000_map_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX"].data[44],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[44],
        fill_val=np.nan,
        real_val=0,
    )

    dap_hdelta_A_map_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX"].data[21],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[21],
        fill_val=np.nan,
        real_val=0,
    )

    dap_hdelta_F_map_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX"].data[23],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[23],
        fill_val=np.nan,
        real_val=0,
    )

    # Correct index for velocity dispersion
    dap_d4000_map_clean_corr = np.sqrt(
        (dap_d4000_map_clean**2) - (dap_gal_fits["SPECINDEX_CORR"].data[43] ** 2)
    )
    dap_dn4000_map_clean_corr = np.sqrt(
        (dap_dn4000_map_clean**2) - (dap_gal_fits["SPECINDEX_CORR"].data[44] ** 2)
    )
    dap_hdelta_A_map_clean_corr = np.sqrt(
        (dap_hdelta_A_map_clean**2) - (dap_gal_fits["SPECINDEX_CORR"].data[21] ** 2)
    )
    dap_hdelta_F_map_clean_corr = np.sqrt(
        (dap_hdelta_F_map_clean**2) - (dap_gal_fits["SPECINDEX_CORR"].data[23] ** 2)
    )

    # Add specindex Maps
    new_hdul.append(fits.ImageHDU(dap_d4000_map_clean_corr, ver=61, name="DAP_D4000"))
    new_hdul.append(fits.ImageHDU(dap_dn4000_map_clean_corr, ver=62, name="DAP_Dn4000"))
    new_hdul.append(
        fits.ImageHDU(dap_hdelta_A_map_clean_corr, ver=63, name="DAP_hdeltaA")
    )
    new_hdul.append(
        fits.ImageHDU(dap_hdelta_F_map_clean_corr, ver=64, name="DAP_hdeltaF")
    )

    # inverse varience specindex maps converted to error values
    dap_d4000_err_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX_IVAR"].data[43],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[43],
        fill_val=np.nan,
        real_val=0,
    )
    dap_dn4000_err_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX_IVAR"].data[44],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[44],
        fill_val=np.nan,
        real_val=0,
    )

    dap_hdelta_A_err_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX_IVAR"].data[21],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[21],
        fill_val=np.nan,
        real_val=0,
    )

    dap_hdelta_F_err_clean = map_masking(
        map_arr=dap_gal_fits["SPECINDEX_IVAR"].data[23],
        mask_arr=dap_gal_fits["SPECINDEX_MASK"].data[23],
        fill_val=np.nan,
        real_val=0,
    )

    # Add Clean error arrays to the fits file
    new_hdul.append(fits.ImageHDU(dap_d4000_err_clean, ver=65, name="DAP_D4000_IVAR"))
    new_hdul.append(fits.ImageHDU(dap_dn4000_err_clean, ver=66, name="DAP_Dn4000_IVAR"))
    new_hdul.append(
        fits.ImageHDU(dap_hdelta_A_err_clean, ver=67, name="DAP_hdeltaA_IVAR")
    )
    new_hdul.append(
        fits.ImageHDU(dap_hdelta_F_err_clean, ver=68, name="DAP_hdeltaF_IVAR")
    )

    # # DQ mask specindex maps
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[51].data[43], ver=69, name="DAP_D4000_DQMASK")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[51].data[44], ver=70, name="DAP_Dn4000_DQMASK")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[51].data[21], ver=71, name="DAP_hdeltaA_DQMASK")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[51].data[23], ver=72, name="DAP_hdeltaF_DQMASK")
    # )

    # # Specindex correction maps (if units are ang DAP_index * DAP_index_CORR)
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[52].data[43], ver=73, name="DAP_D4000_CORR")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[52].data[44], ver=74, name="DAP_Dn4000_CORR")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[52].data[21], ver=75, name="DAP_hdeltaA_CORR")
    # )
    # new_hdul.append(
    #     fits.ImageHDU(dap_gal_fits[52].data[23], ver=76, name="DAP_hdeltaF_CORR")
    # )

    # Radius in kpc
    new_hdul.append(fits.ImageHDU(r_map * spaxel_diamter_in_kpc, ver=69, name="R_KPC"))

    ######## Gas velocity dispersion measurments from the MaNGA DAP files
    # - None corrected gas velocity dispersion - Remove masked pixels from array
    ha_sigma_gas_map_clean = map_masking(
        map_arr=dap_gal_fits["EMLINE_GSIGMA"].data[23],
        mask_arr=dap_gal_fits["EMLINE_GSIGMA_MASK"].data[23],
        fill_val=np.nan,
        real_val=0,
    )

    # - Corrected velocity dispersion for instrunmental resolution effects
    ha_sigma_gas_corr = np.sqrt(
        (ha_sigma_gas_map_clean**2) - (dap_gal_fits["EMLINE_INSTSIGMA"].data[23] ** 2)
    )

    ha_sigma_gas_map_error = map_masking(
        map_arr=dap_gal_fits["EMLINE_GSIGMA_IVAR"].data[23],
        mask_arr=dap_gal_fits["EMLINE_GSIGMA_MASK"].data[23],
        fill_val=np.nan,
        real_val=0,
    )
    # - Add to mm_fits file
    new_hdul.append(
        fits.ImageHDU(ha_sigma_gas_corr, ver=70, name="GASVEL_DISP")
    )  # [km/s]
    new_hdul.append(
        fits.ImageHDU(ha_sigma_gas_map_error, ver=71, name="GASVEL_ERR_DISP")
    )  # [km/s]

    ######## Stellar velocity dispersion measurments from the MaNGA DAP files
    # Non-corrected velcoity dispersion
    ha_sigma_star_map_clean = map_masking(
        map_arr=dap_gal_fits["STELLAR_SIGMA"].data,
        mask_arr=dap_gal_fits["STELLAR_SIGMA_MASK"].data,
        fill_val=np.nan,
        real_val=0,
    )
    # Error
    ha_sigma_star_map_error = map_masking(
        map_arr=dap_gal_fits["STELLAR_SIGMA_IVAR"].data,
        mask_arr=dap_gal_fits["STELLAR_SIGMA_MASK"].data,
        fill_val=np.nan,
        real_val=0,
    )
    # - Corrected velocity dispersion for instrunmental
    ha_sigma_star_corr1 = np.sqrt(
        (ha_sigma_star_map_clean**2)
        - (dap_gal_fits["STELLAR_SIGMACORR"].data[0] ** 2)
    )
    ha_sigma_star_corr2 = np.sqrt(
        (ha_sigma_star_map_clean**2)
        - (dap_gal_fits["STELLAR_SIGMACORR"].data[1] ** 2)
    )

    # - Add to new stellar velocity maps corrected
    new_hdul.append(fits.ImageHDU(ha_sigma_star_corr1, ver=72, name="STARVEL_DISP1"))
    new_hdul.append(fits.ImageHDU(ha_sigma_star_corr2, ver=73, name="STARVEL_DISP2"))

    # Sigma
    new_hdul.append(fits.ImageHDU(ha_sigma_star_corr2, ver=73, name="STARVEL_DISP2"))

    new_hdul.append(
        fits.ImageHDU(ha_sigma_star_map_error, ver=74, name="STARVEL_DISPERR")
    )

    # Morphology data from galaxy Zoo
    # Update header
    prihdr = new_hdul[0].header
    prihdr["SPX_AREA"] = "{} Kpc^2".format(str(spaxel_area_kpc)[:5])
    prihdr["kpc_arcs"] = "{}".format(np.round(spaxel_diamter_in_kpc, 4))
    # prihdr["kpc/arcsec"] = "{}".format(str(spaxel_diamter_in_kpc)[:5])
    prihdr["REFF"] = "{} ARCSECS".format(str(r_eff)[:5])
    prihdr["z"] = "{} redshift".format(str(z)[:5])
    prihdr["plateifu"] = plateifu

    # Visual classification on T-Type morphology for BBRD galaxies
    if (
        plateifu == "10001-3702"
        or plateifu == "11758-1901"
        or plateifu == "11827-1902"
        or plateifu == "8254-1902"
        or plateifu == "8565-1902"
    ):
        prihdr["TTYPE_MM"] = "S0"

    elif plateifu == "10217-6103" or plateifu == "8312-12704":
        prihdr["TTYPE_MM"] = "SBa"

    elif plateifu == "12094-1901" or plateifu == "8550-12703":
        prihdr["TTYPE_MM"] = "SBb"

    elif plateifu == "9894-3703" or plateifu == "8465-1902":
        prihdr["TTYPE_MM"] = "Sa"

    elif plateifu == "8595-3703":
        prihdr["TTYPE_MM"] = "Merger"

    # Store integrated measurments from Pipe3D SDSS17Pipe3D_v3_1_1.csv

    # Write data to FITS file
    new_hdul.writeto(
        f"/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/{sample}_MMfits/{plateifu}_MM.fits",
        overwrite=True,
    )


# Read FITS file extension
def read_fits_ext(fits_file, ext=1):
    hdu = fits.open(fits_file)
    sci_maps = hdu[ext].data
    hdu.close()
    return sci_maps


# ----------------------------------------Run for single fits file (for testing)--------------------------------------
# Run code for a single file

# plateifu, nsa_z = "10001-3702", 0.0256063
# write_maps2fits_mpldap(plateifu, mode="local", sample="bbrd", z=nsa_z)
# ------------------------------------------------------------------------------------------------


# -----------------------------------Execution of function-------------------------------------------------------------
# Read bbrd final crossmatch table
bbrd_df = pd.read_csv(
    "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/global_merged_dat/bbrd_global_db.csv"
)
# Run code for all galaxies in a sample
print("Executing gernerate functions to make new MM_FITS files")
for plateifu, nsa_z in zip(bbrd_df["plateifu"], bbrd_df["nsa_z"]):
    # print(plateifu, nsa_z)
    write_maps2fits_mpldap(plateifu, mode="local", sample="bbrd", z=nsa_z)


# --------------------------Storing the 2D maps in ID series of Datafame--------------------------------------

# 1. Flatten 2D map to a 1D column and stores the in a CSV file
# 2. Store 2D Maps as 1D columns in a pandas dataframe in organized by the R/Reff value

# List of BBRD FITS files made by generate code MM
bbrd_fits_filelist = glob.glob(
    "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/*.fits"
)

# The range of FITS extension np.arange(1,1+last ver #,1)
extnum_list = np.arange(1, 75, 1)

# 1.
for fit in bbrd_fits_filelist:
    hdu = fits.open(fit)
    ifu_df = pd.DataFrame()
    for extnum in extnum_list:
        # Read in 2d data from fits file extension
        map2d_data = read_fits_ext(fit, ext=hdu[extnum - 1].name)
        # Flatten 2d map
        map1d_data = map2d_data.flatten(order="C")
        # print(extnum, hdu[extnum-1].ver, hdu[extnum-1].name, map1d_data.shape)
        # Pair extname column with the 1D Map data
        ifu_df[hdu[extnum - 1].name] = map1d_data

    # Add a series of plateifu to keep the data labeled
    ifu_df["PLATEIFU"] = pd.Series(hdu[0].header["plateifu"], index=range(len(ifu_df)))
    ifu_df.to_csv(
        path_or_buf="/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/{}_map.csv".format(
            hdu[0].header["plateifu"]
        ),
        sep=",",
    )
    hdu.close()


# LG12
# -----------------------------------Execution of function-------------------------------------------------------------
# Read lg12 final crossmatch table
lg12_df = pd.read_csv(
    "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/global_merged_dat/lg12_global_db.csv"
)
# Run code for all galaxies in a sample
print("Executing gernerate functions to make new MM_FITS files")
for plateifu, nsa_z in zip(lg12_df["plateifu"], lg12_df["nsa_z"]):
    # print(plateifu, nsa_z)
    write_maps2fits_mpldap(plateifu, mode="local", sample="lg12", z=nsa_z)


# --------------------------Storing the 2D maps in ID series of Datafame--------------------------------------

# 1. Flatten 2D map to a 1D column and stores the in a CSV file
# 2. Store 2D Maps as 1D columns in a pandas dataframe in organized by the R/Reff value

# List of BBRD FITS files made by generate code MM
lg12_fits_filelist = glob.glob(
    "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/lg12_MMfits/*.fits"
)

# The range of FITS extension np.arange(1,1+last ver #,1)
extnum_list = np.arange(1, 75, 1)

# 1.
for fit in lg12_fits_filelist:
    hdu = fits.open(fit)
    ifu_df = pd.DataFrame()
    for extnum in extnum_list:
        # Read in 2d data from fits file extension
        map2d_data = read_fits_ext(fit, ext=hdu[extnum - 1].name)
        # Flatten 2d map
        map1d_data = map2d_data.flatten(order="C")
        # print(extnum, hdu[extnum-1].ver, hdu[extnum-1].name, map1d_data.shape)
        # Pair extname column with the 1D Map data
        ifu_df[hdu[extnum - 1].name] = map1d_data

    # print(ifu_df.shape)
    ifu_df.to_csv(
        path_or_buf="/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/lg12_MMfits/{}_map.csv".format(
            hdu[0].header["plateifu"]
        ),
        sep=",",
    )
    hdu.close()
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------

# # ------------- Seperates the files into groups that I set manually and make supercsv file for each group

# # Combine all dataframes into a super dataframe
# bbrd_csv_list = glob.glob(
#     "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/*_map.csv"
# )
# combo_csv_list = []
# agn_combo_csv_list = []
# xl_combo_csv_list = []
# for bbrd in bbrd_csv_list:
#     # Remove the merger and AGN galaxy from the Combo csv file and adds AGn to there own csv
#     if (
#         bbrd == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/9183-3703_map.csv"
#         or bbrd
#         == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/11827-1902_map.csv"
#     ):
#         # print("appending AGN")
#         bbrd_df = pd.read_csv(bbrd)
#         agn_combo_csv_list.append(bbrd_df)

#     elif bbrd == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/8595-3703_map.csv":
#         pass

#     else:
#         bbrd_df = pd.read_csv(bbrd)
#         # print(bbrd, bbrd_df.shape)
#         combo_csv_list_df.append(bbrd_df)

# # Remove large galaxies from the combo_bbrd and and adds to there own CSV files
# elif (bbrd == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/8312-12704_map.csv"
# or bbrd == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/8465-9102_map.csv"
# or bbrd == "/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/8550-12703_map.csv"
# ):
#     print('Appending XL galaxies CSV')
#     bbrd_df = pd.read_csv(bbrd)
#     print(bbrd, bbrd_df.shape)
#     xl_combo_csv_list.append(bbrd_df)

# # Combine csv files
# bbrd_supercsv = pd.concat(combo_csv_list, ignore_index=True)
# # Replace inf value with nans
# bbrd_supercsv_clean = bbrd_supercsv.replace([np.inf, -np.inf], np.nan)
# bbrd_supercsv_clean.to_csv(
#     path_or_buf="/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/supercsv_ifu_df/bbrd_combo.csv",
#     sep=",",
# )

# # Combine AGN csv files
# bbrd_agn_supercsv = pd.concat(agn_combo_csv_list, ignore_index=True)
# # Replace inf value with nans
# bbrd_agn_supercsv_clean = bbrd_agn_supercsv.replace([np.inf, -np.inf], np.nan)
# bbrd_agn_supercsv_clean.to_csv(
#     path_or_buf="/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/supercsv_ifu_df/bbrd_agn_combo.csv",
#     sep=",",
# )

# bbrd_xl_supercsv = pd.concat(xl_combo_csv_list)
# # Replace inf value with nans
# bbrd_xl_supercsv_clean = bbrd_xl_supercsv.replace([np.inf, -np.inf], np.nan)
# bbrd_xl_supercsv_clean.to_csv(
#     path_or_buf="/Users/mmckay/Desktop/research/BreakBRD_LG12_analysis_repo/2d_bbrd_lg12_maps_v5/bbrd_MMfits/bbrd_xl_combo.csv",
#     sep=",",
# )


# ----------------------------------------LG12 SAMPLE--------------------------------------------------------


# # ------------- Seperates the files into groups that I set manually and make supercsv file for each group

# # Combine all dataframes into a super dataframe
# lg12_csv_list = glob.glob(
#     "/Users/mmckay/Desktop/research/FMR_MZR/lg12_MMfits/*_map.csv"
# )
# combo_csv_list = []
# for lg12 in lg12_csv_list:
#     lg12_df = pd.read_csv(lg12)
#     print(lg12, lg12_df.shape)
#     combo_csv_list.append(lg12_df)


# # Write the Combined as a super-csv file
# lg12_supercsv = pd.concat(combo_csv_list)
# # Replace inf value with nans
# lg12_supercsv_clean = lg12_supercsv.replace([np.inf, -np.inf], np.nan)
# lg12_supercsv_clean.to_csv(
#     path_or_buf="/Users/mmckay/Desktop/research/FMR_MZR/lg12_MMfits/lg12_combo.csv",
#     sep=",",
# )
