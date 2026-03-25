use dist_corr::{DistCorrelation, DistCovariance};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn map_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
#[pyo3(signature = (v1, v2, v1_binary=false, v2_binary=false))]
fn distance_correlation(
    v1: PyReadonlyArray1<'_, f64>,
    v2: PyReadonlyArray1<'_, f64>,
    v1_binary: bool,
    v2_binary: bool,
) -> PyResult<f64> {
    let v1 = v1.as_slice().map_err(map_err)?;
    let v2 = v2.as_slice().map_err(map_err)?;

    DistCorrelation
        .compute_binary(v1, v2, v1_binary, v2_binary)
        .map_err(map_err)
}

#[pyfunction]
#[pyo3(signature = (v1, v2, v1_binary=false, v2_binary=false))]
fn distance_covariance(
    v1: PyReadonlyArray1<'_, f64>,
    v2: PyReadonlyArray1<'_, f64>,
    v1_binary: bool,
    v2_binary: bool,
) -> PyResult<f64> {
    let v1 = v1.as_slice().map_err(map_err)?;
    let v2 = v2.as_slice().map_err(map_err)?;

    DistCovariance
        .compute_binary(v1, v2, v1_binary, v2_binary)
        .map_err(map_err)
}

#[pyfunction]
fn distance_variance(v: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let v = v.as_slice().map_err(map_err)?;
    DistCovariance.compute_var(v).map_err(map_err)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(distance_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(distance_variance, m)?)?;
    Ok(())
}
