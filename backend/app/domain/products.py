"""Static product facts: which policy document governs each product and which incidents it can cover."""

from app.domain.enums import IncidentType, ProductType

# File names of the policy wordings indexed into Qdrant in Phase 4; citations reference these exact names.
POLICY_DOCUMENTS: dict[ProductType, str] = {
    ProductType.MOTOR: "Motor_Policy.pdf",
    ProductType.HEALTH: "Health_Policy.pdf",
    ProductType.HOME: "Home_Policy.pdf",
}

INCIDENT_TYPES_BY_PRODUCT: dict[ProductType, tuple[IncidentType, ...]] = {
    ProductType.MOTOR: (
        IncidentType.COLLISION,
        IncidentType.THEFT,
        IncidentType.VANDALISM,
        IncidentType.NATURAL_CALAMITY,
    ),
    ProductType.HEALTH: (IncidentType.HOSPITALISATION, IncidentType.SURGERY, IncidentType.DAY_CARE),
    ProductType.HOME: (
        IncidentType.FIRE,
        IncidentType.BURGLARY,
        IncidentType.WATER_DAMAGE,
        IncidentType.NATURAL_CALAMITY,
    ),
}
