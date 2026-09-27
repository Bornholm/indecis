package main

import "github.com/bornholm/indecis"

// classic est la liste de catégories de la preuve de concept : un tri
// classique de boîte de réception d'organisation. Elle ne sert jamais à
// l'entraînement, pour mesurer ce que vaut une liste inconnue du modèle.
var classic = []indecis.Candidate{
	{Name: "Support technique", Description: "Panne, bug, problème d'accès ou demande d'aide sur un outil, un logiciel ou un service informatique."},
	{Name: "Facturation et paiement", Description: "Factures, paiements, relances, remboursements, questions de tarif sur une prestation déjà vendue."},
	{Name: "Commande et livraison", Description: "Suivi de commande, expédition, retard de livraison, retour ou échange de produit."},
	{Name: "Réclamation", Description: "Mécontentement exprimé ou plainte formelle d'un client ou d'un partenaire."},
	{Name: "Demande commerciale", Description: "Demande de devis, de tarifs ou d'informations avant achat, proposition commerciale, prospection."},
	{Name: "Réunion et agenda", Description: "Invitation, planification, report ou annulation de réunion, compte rendu de réunion."},
	{Name: "Ressources humaines", Description: "Recrutement, candidature, congés, paie, formation, contrat de travail, vie des salariés."},
	{Name: "Juridique et contrats", Description: "Contrats, accords, clauses, conformité réglementaire, litiges, avis juridiques."},
	{Name: "Suivi de projet", Description: "Échanges de travail sur un projet ou une activité : avancement, rapports, chiffres, documents à relire."},
	{Name: "Newsletter et marketing", Description: "Lettre d'information, promotion, annonce ou communication envoyée à une liste de diffusion."},
	{Name: "Notification automatique", Description: "Message généré par un système : alerte, confirmation, accusé de réception, rappel automatique."},
	{Name: "Personnel", Description: "Échange privé, non professionnel : famille, amis, loisirs."},
	{Name: "Spam et hameçonnage", Description: "Message non sollicité, publicité abusive ou tentative de fraude."},
}
