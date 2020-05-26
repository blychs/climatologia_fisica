import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
# import metpy
import metpy.units as units
import metpy.calc as mpcalc
# import metpy.constants as mpconstants
# import seaborn as sns
import cartopy.feature as cfeature
# from cartopy.util import add_cyclic_point


def xradd_cyclic_point(xarray_obj, dim, period=None):
    if period is None:
        period = ((xarray_obj.sizes[dim] *
                   xarray_obj.coords[dim][:2]).diff(dim).item())
    first_point = xarray_obj.isel({dim: slice(1)})
    first_point.coords[dim] = first_point.coords[dim]+period
    return xr.concat([xarray_obj, first_point], dim=dim)


def load_file(path):
    """
    Carga archivo/s y devuelve:
    Ensamble, ensamble en NDJFM, ensamble en MJJAS, media, media en NDJFM,
    media en MJJAS"""
    file_dict = {}
    ds = xr.open_mfdataset(path, combine='nested',
                           concat_dim='ensemble').mean(dim='ensemble')
    file_dict['ds'] = xradd_cyclic_point(ds, 'lon')
    file_dict['mean'] = xradd_cyclic_point(ds.mean(dim='time'), 'lon')
    return file_dict


def ploteo_general(dataarray, title=None, vmin=None, vmax=None,
                   projection=ccrs.PlateCarree(), figsize=(20, 10),
                   extend='max', cmap=None, under='none', over='none'):
    plt.figure(figsize=(figsize))
    ax = plt.axes(projection=projection)
    if cmap is not None:
        ploteo = dataarray.plot(vmin=vmin, vmax=vmax, extend=extend, ax=ax,
                                cmap=cmap)
    else:
        ploteo = dataarray.plot(vmin=vmin, vmax=vmax, extend=extend, ax=ax,
                                transform=ccrs.PlateCarree())
    cmap = ploteo.get_cmap()
    if under != 'none':
        cmap.set_under(under)
    if over != 'none':
        cmap.set_over(over)
    ax.coastlines()
    plt.title(title)
    ax.add_feature(cfeature.BORDERS)
    return ax


def latlon_domain(dataset, lats_lons):
    """
    Aplica dominio de latitudes y longitudes a un xr.dataset.
    Debe recibir una lista con lon1, lon2, lat1, lat2"""
    dataoutput = dataset.loc[dict(lon=slice(lats_lons[0], lats_lons[1]),
                                  lat=slice(lats_lons[2], lats_lons[3]))]
    return dataoutput


def calculate_q(rhum, pressure, temperature):

    """ Calcula q desde relative humidity, pressure
    y temperature, La temperatura DEBE estar en Celsius
    segun https://pielkeclimatesci.wordpress.com/2010/07/22/
      guest-post-calculating-moist-enthalpy-from-
      usual-meteorological-measurements-by-francis-massen/"""

    saturation_vapor_pres = 10 ** ((0.7859 + 0.03477 * temperature /
                                    (1 + 0.00412*temperature)) + 2)
    vapor_pres = rhum / 100 * saturation_vapor_pres

    # Presión en pascales
    return (0.622/((pressure * 100 / vapor_pres) - 0.378))


def calculate_h(tas_ds, huss_ds):

    """Calculate moist static energy"""

    heights = 2 * np.ones((tas_ds.dims['lat'],
                           tas_ds.dims['lon'])) * units.meter

    tas_ds.tas.attrs['units'] = 'kelvin'
    temperature = tas_ds.metpy.parse_cf('tas')
    huss_ds.huss.attrs['units'] = 'dimensionless'
    humidity = huss_ds.metpy.parse_cf('huss')

    m_s_e = mpcalc.moist_static_energy(heights=heights,
                                       temperature=temperature,
                                       specific_humidity=humidity).to('J/kg')
    return m_s_e


def advection(variable, u_da, v_da, units_wind='m/s'):
    """
    Calcula la advección en base a un dataarray de u y un dataarray de v
    en m/s """
    lats = u_da.lat
    lons = u_da.lon
    u = u_da.values * units['m/s']
    v = v_da.values * units['m/s']
    dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
    return mpcalc.advection(variable, [u, v],
                            (dx, dy), dim_order='yx')


def streamQuiver(ax, sp, *args, spacing=None, n=5,**kwargs):
    """ Plot arrows from streamplot data
    The number of arrows per streamline is controlled either by `spacing` or by `n`.
    See `lines_to_arrows`.
    """
    def curve_coord(line=None):
        """ return curvilinear coordinate """
        x=line[:,0]
        y=line[:,1]
        s     = np.zeros(x.shape)
        s[1:] = np.sqrt((x[1:]-x[0:-1])**2+ (y[1:]-y[0:-1])**2)
        s     = np.cumsum(s)
        return s

    def curve_extract(line,spacing,offset=None):
        """ Extract points at equidistant space along a curve"""
        x=line[:,0]
        y=line[:,1]
        if offset is None:
            offset=spacing/2
        # Computing curvilinear length
        s = curve_coord(line)
        offset=np.mod(offset,s[-1]) # making sure we always get one point
        # New (equidistant) curvilinear coordinate
        sExtract=np.arange(offset,s[-1],spacing)
        # Interpolating based on new curvilinear coordinate
        xx=np.interp(sExtract,s,x);
        yy=np.interp(sExtract,s,y);
        return np.array([xx,yy]).T

    def seg_to_lines(seg):
        """ Convert a list of segments to a list of lines """
        def extract_continuous(i):
            x=[]
            y=[]
            # Special case, we have only 1 segment remaining:
            if i==len(seg)-1:
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                x.append(seg[i][1,0])
                y.append(seg[i][1,1])
                return i,x,y
            # Looping on continuous segment
            while i<len(seg)-1:
                # Adding our start point
                x.append(seg[i][0,0])
                y.append(seg[i][0,1])
                # Checking whether next segment continues our line
                Continuous= all(seg[i][1,:]==seg[i+1][0,:])
                if not Continuous:
                    # We add our end point then
                    x.append(seg[i][1,0])
                    y.append(seg[i][1,1])
                    break
                elif i==len(seg)-2:
                    # we add the last segment
                    x.append(seg[i+1][0,0])
                    y.append(seg[i+1][0,1])
                    x.append(seg[i+1][1,0])
                    y.append(seg[i+1][1,1])
                i=i+1
            return i,x,y
        lines=[]
        i=0
        while i<len(seg):
            iEnd,x,y=extract_continuous(i)
            lines.append(np.array( [x,y] ).T)
            i=iEnd+1
        return lines

    def lines_to_arrows(lines,n=5,spacing=None,normalize=True):
        """ Extract "streamlines" arrows from a set of lines
        Either: `n` arrows per line
            or an arrow every `spacing` distance
        If `normalize` is true, the arrows have a unit length
        """
        if spacing is None:
            # if n is provided we estimate the spacing based on each curve lenght)
            spacing = [ curve_coord(l)[-1]/n for l in lines]
        try:
            len(spacing)
        except:
            spacing=[spacing]*len(lines)

        lines_s=[curve_extract(l,spacing=sp,offset=sp/2)         for l,sp in zip(lines,spacing)]
        lines_e=[curve_extract(l,spacing=sp,offset=sp/2+0.01*sp) for l,sp in zip(lines,spacing)]
        arrow_x  = [l[i,0] for l in lines_s for i in range(len(l))]
        arrow_y  = [l[i,1] for l in lines_s for i in range(len(l))]
        arrow_dx = [le[i,0]-ls[i,0] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]
        arrow_dy = [le[i,1]-ls[i,1] for ls,le in zip(lines_s,lines_e) for i in range(len(ls))]

        if normalize:
            dn = [ np.sqrt(ddx**2 + ddy**2) for ddx,ddy in zip(arrow_dx,arrow_dy)]
            arrow_dx = [ddx/ddn for ddx,ddn in zip(arrow_dx,dn)]
            arrow_dy = [ddy/ddn for ddy,ddn in zip(arrow_dy,dn)]
        return  arrow_x,arrow_y,arrow_dx,arrow_dy

    # --- Main body of streamQuiver
    # Extracting lines
    seg   = sp.lines.get_segments() # list of (2, 2) numpy arrays
    lines = seg_to_lines(seg)       # list of (N,2) numpy arrays
    # Convert lines to arrows
    ar_x, ar_y, ar_dx, ar_dy = lines_to_arrows(lines,spacing=spacing,n=n,normalize=True)
    # Plot arrows
    qv=ax.quiver(ar_x, ar_y, ar_dx, ar_dy, *args, angles='xy', **kwargs)
    return qv

